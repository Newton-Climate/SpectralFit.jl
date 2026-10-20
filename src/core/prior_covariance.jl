# Prior covariances with vertical correlations within each gas profile.

"""
    correlated_prior_covariance(sigma::ComponentVector, coordinate, correlation_lengths::NamedTuple;
                                nugget = 1e-6) -> Matrix{Float64}

Prior covariance Sa for prior standard deviations `sigma` (state units) with Gaussian
correlations between the layers of each gas named in `correlation_lengths`:

    Sa[i, j] = sigma_i sigma_j (exp(-(c_i - c_j)^2 / (2 L^2)) + nugget * (i == j))

`coordinate` holds one value per layer (height in km, or pressure in hPa) and `L` is the
gas's correlation length in the same units. Other entries (baseline coefficients, gases not
named) are uncorrelated.

The nugget (dimensionless) is needed because the Gaussian kernel is numerically singular
once `L` is more than about twice the layer spacing (50 hPa on layers 10 to 30 hPa apart
gives a condition number near 1e16). With the default 1e-6 the condition number of each gas
block is at most about `n_layers / nugget`, and the added uncorrelated variance is 1e-6 of
`sigma^2`.

```julia
sigma = ComponentVector(H2O = fill(1e21, 10), CH4 = fill(1e18, 10), baseline = (window_1 = ones(3),))
Sa = correlated_prior_covariance(sigma, range(0, 9, length = 10), (CH4 = 2.0,))   # km
```
"""
function correlated_prior_covariance(sigma::ComponentVector, coordinate::AbstractVector,
                                     correlation_lengths::NamedTuple; nugget::Real = 1e-6)
    nugget >= 0 || throw(ArgumentError("nugget: must be non-negative, got $nugget"))
    s = collect(Float64, sigma)
    all(v -> isfinite(v) && v > 0, s) ||
        throw(ArgumentError("sigma: every standard deviation must be positive and finite"))
    Sa = Matrix(Diagonal(s .^ 2))
    for (gas, L) in pairs(correlation_lengths)
        L > 0 || throw(ArgumentError("correlation_lengths.$gas: must be positive, got $L"))
        idx = state_indices(sigma, gas)
        length(idx) == length(coordinate) || throw(DimensionMismatch(
            "correlation_lengths.$gas: $gas has $(length(idx)) layers, the coordinate has $(length(coordinate))"))
        for (a, i) in enumerate(idx), (b, j) in enumerate(idx)
            Sa[i, j] = s[i] * s[j] * (exp(-(coordinate[a] - coordinate[b])^2 / (2L^2)) + nugget * (i == j))
        end
    end
    return Sa
end

"""
    correlated_prior_precision(sigma, coordinate, correlation_lengths; nugget = 1e-6) -> Matrix{Float64}

Inverse of `correlated_prior_covariance`, computed in sigma-scaled form (Sa = D C D with C
the correlation matrix, so Sa^-1 = D^-1 C^-1 D^-1) by Cholesky, and symmetrized. Use it as
the `prior_precision` of a `RetrievalProblem`.

```julia
P = correlated_prior_precision(sigma, z_km, (H2O = 0.01, CH4 = 0.5))
problem = RetrievalProblem(model, y, obs_precision(LogSpace(), m), x_prior, P)
```
"""
function correlated_prior_precision(sigma::ComponentVector, coordinate::AbstractVector,
                                    correlation_lengths::NamedTuple; nugget::Real = 1e-6)
    Sa = correlated_prior_covariance(sigma, coordinate, correlation_lengths; nugget = nugget)
    d = 1 ./ collect(Float64, sigma)
    C = cholesky(Symmetric(d .* Sa .* d'); check = false)
    issuccess(C) || throw(ArgumentError(
        "correlated_prior_precision: the correlation matrix is not positive definite; increase nugget (now $nugget)"))
    P = d .* inv(C) .* d'
    return Matrix{Float64}((P + P') ./ 2)
end

export correlated_prior_covariance, correlated_prior_precision
