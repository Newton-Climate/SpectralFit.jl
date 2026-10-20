# Optimal-estimation diagnostics (section 3.11 of docs/architecture_plan.md).
#
# Every function here takes plain matrices and vectors, so it can be checked against closed
# forms. Notation (Rodgers 2000): K is the Jacobian (n_obs x n_state), obs_precision is the
# inverse observation covariance Se_inv (n_obs x n_obs), prior_precision is the inverse prior
# covariance Sa_inv (n_state x n_state). Units follow the state and the measurement space:
# K has units of measurement per state unit, the posterior covariance has state units squared.
#
# The one place that turns these into a result is `retrieval_result` in retrieval_result.jl;
# no solver computes a posterior itself (issue 1).
#
# `obs_precision` may be any symmetric positive definite matrix (dense, `Diagonal`, sparse) or
# a `CovariancePrecision`, which holds the factorized covariance and is applied by solves.

"""Accepted observation precisions: an `AbstractMatrix` or a `CovariancePrecision`."""
const ObservationPrecision = Union{AbstractMatrix, CovariancePrecision}

"""
    normal_matrix(K, obs_precision, prior_precision) -> Symmetric{Float64, Matrix{Float64}}

The Gauss-Newton normal matrix `K' * obs_precision * K + prior_precision`, the inverse of the
posterior covariance, as a dense symmetric matrix (state units to the power -2).

```julia
N = normal_matrix(K, Diagonal(fill(1e4, size(K, 1))), Diagonal([1.0, 1.0]))
```
"""
function normal_matrix(K::AbstractMatrix, obs_precision::ObservationPrecision,
                       prior_precision::AbstractMatrix)
    information = Matrix{Float64}(K' * (obs_precision * K))
    N = information + prior_precision
    return Symmetric((N + N') ./ 2)
end

"""
    posterior_covariance(K, obs_precision, prior_precision) -> Matrix{Float64}

Posterior covariance `S = inv(K' * obs_precision * K + prior_precision)` (Rodgers eq. 2.27), in
state units squared. Always includes the prior term; with a zero prior precision it is the
least-squares covariance `inv(K' * obs_precision * K)`.

```julia
K = [1.0 0.0; 0.0 2.0; 1.0 1.0]
S = posterior_covariance(K, Diagonal(fill(100.0, 3)), Diagonal([1.0, 1.0]))
```
"""
posterior_covariance(K::AbstractMatrix, obs_precision::ObservationPrecision,
                     prior_precision::AbstractMatrix) =
    scaled_inverse(normal_matrix(K, obs_precision, prior_precision))

"""
    scaled_inverse(N::Symmetric) -> Matrix{Float64}

Inverse of a symmetric positive definite matrix, computed as `s .* inv(M) .* s'` with
`M = s .* N .* s'` scaled to unit diagonal (`s = 1 ./ sqrt.(diag(N))`) and inverted by
Cholesky. State vectors mix units (columns near 1e22 molecules/cm2 next to baseline
coefficients near 1), so the unscaled normal matrix can have a condition number near 1e54
while the scaled one is moderate; inverting it unscaled loses the small entries. Throws a
`PosDefException` when `N` is not positive definite.
"""
function scaled_inverse(N::Symmetric)
    d = diag(N)
    all(>(0), d) || throw(PosDefException(findfirst(<=(0), d)))
    s = 1 ./ sqrt.(d)
    M = Symmetric(Matrix{Float64}(s .* N .* s'))
    Minv = inv(cholesky(M))
    S = s .* Minv .* s'
    return Matrix{Float64}((S + S') ./ 2)
end

"""
    gain_matrix(K, obs_precision, prior_precision) -> Matrix{Float64}

Gain matrix `G = S * K' * obs_precision` (n_state x n_obs), the sensitivity of the retrieved
state to the measurement (state units per measurement unit).

```julia
G = gain_matrix(K, Se_inv, Sa_inv)
dx = G * (y - y_model)
```
"""
gain_matrix(K::AbstractMatrix, obs_precision::ObservationPrecision, prior_precision::AbstractMatrix) =
    posterior_covariance(K, obs_precision, prior_precision) *
    Matrix{Float64}(_adjoint_times_precision(K, obs_precision))

"""
    averaging_kernel(K, obs_precision, prior_precision) -> Matrix{Float64}

Averaging kernel `A = S * K' * obs_precision * K = G * K` (n_state x n_state, dimensionless
for each pair of elements in the same units), equal to `I - S * prior_precision`. With a zero
prior precision it is the identity.

```julia
A = averaging_kernel(K, Se_inv, Sa_inv)
dfs_total = tr(A)
```
"""
function averaging_kernel(K::AbstractMatrix, obs_precision::ObservationPrecision,
                          prior_precision::AbstractMatrix)
    S = posterior_covariance(K, obs_precision, prior_precision)
    return S * Matrix{Float64}(K' * (obs_precision * K))
end

"""
    dfs_per_element(averaging_kernel) -> Vector{Float64}

Degrees of freedom for signal per state element: the diagonal of the averaging kernel.

```julia
dfs = dfs_per_element(averaging_kernel(K, Se_inv, Sa_inv))
```
"""
dfs_per_element(A::AbstractMatrix) = Vector{Float64}(diag(A))

"""
    dfs_total(averaging_kernel) -> Float64

Total degrees of freedom for signal: the trace of the averaging kernel.
"""
dfs_total(A::AbstractMatrix) = Float64(tr(A))

"""
    rodgers_d2(dx, K, obs_precision, prior_precision) -> Float64

Rodgers convergence statistic `d2 = dx' * (K' * obs_precision * K + prior_precision) * dx`
(Rodgers eq. 5.29) for a state step `dx`, dimensionless. Solvers divide it by `n_state` and
compare with `rodgers_tolerance`.

```julia
rodgers_d2([0.1, 0.0], K, Se_inv, Sa_inv)
```
"""
rodgers_d2(dx::AbstractVector, K::AbstractMatrix, obs_precision::ObservationPrecision,
           prior_precision::AbstractMatrix) =
    rodgers_d2(dx, normal_matrix(K, obs_precision, prior_precision))

"""
    rodgers_d2(dx, normal_matrix) -> Float64

Same statistic from a precomputed `normal_matrix`.
"""
rodgers_d2(dx::AbstractVector, N::AbstractMatrix) = Float64(dot(dx, N * dx))

"""
    observation_cost(y, y_model, obs_precision) -> Float64

Observation term of the cost, `(y - y_model)' * obs_precision * (y - y_model)`, dimensionless.

```julia
observation_cost([1.0, 2.0], [1.1, 1.9], Diagonal([100.0, 100.0]))   # 2.0
```
"""
function observation_cost(y::AbstractVector, y_model::AbstractVector, obs_precision::ObservationPrecision)
    r = y - y_model
    return Float64(dot(r, obs_precision * r))
end

"""
    prior_cost(x, x_prior, prior_precision) -> Float64

Prior term of the cost, `(x - x_prior)' * prior_precision * (x - x_prior)`, dimensionless.
"""
function prior_cost(x::AbstractVector, x_prior::AbstractVector, prior_precision::AbstractMatrix)
    d = collect(x) - collect(x_prior)
    return Float64(dot(d, prior_precision * d))
end

"""
    total_cost(y, y_model, obs_precision, x, x_prior, prior_precision) -> Float64

Full maximum a posteriori cost: `observation_cost + prior_cost`.
"""
total_cost(y, y_model, obs_precision, x, x_prior, prior_precision) =
    observation_cost(y, y_model, obs_precision) + prior_cost(x, x_prior, prior_precision)

"""
    chi2_per_dof(cost, n_obs, n_state) -> Float64

Chi squared per degree of freedom, `cost / (n_obs - n_state)`. Returns `NaN` when
`n_obs <= n_state`. For a correct noise model and a weak prior its expected value is 1.

```julia
chi2_per_dof(17.3, 20, 3)   # 1.0176...
```
"""
function chi2_per_dof(cost::Real, n_obs::Integer, n_state::Integer)
    dof = n_obs - n_state
    return dof > 0 ? Float64(cost) / dof : NaN
end

"""
    full_weight_observations(obs_precision) -> BitVector

Observations at full weight: for a diagonal precision, those whose precision is at least 1e-3
of the median (observations in masked windows are de-weighted by `mask_factor`, 1e-5 by
default); for any other precision, all observations.
"""
function full_weight_observations(P::Diagonal)
    d = P.diag
    return d .>= 1e-3 * median(d)
end
full_weight_observations(P) = trues(size(P, 1))

"""
    residual_rmse(y, y_model, obs_precision) -> Float64

Root-mean-square residual `sqrt(mean((y - y_model)^2))` in measurement-space units, over the
observations at full weight (`full_weight_observations`). A goodness-of-fit metric in data
units, independent of the noise model, unlike chi squared.

```julia
residual_rmse(result.y, result.y_model, result.obs_precision) == result.rmse
```
"""
function residual_rmse(y::AbstractVector, y_model::AbstractVector, obs_precision)
    keep = full_weight_observations(obs_precision)
    any(keep) || return NaN
    r = (y .- y_model)[keep]
    return all(isfinite, r) ? sqrt(sum(abs2, r) / length(r)) : NaN
end

"""
    prior_influence(dfs_total, n_state) -> Float64

Fraction of the state determined by the prior: `1 - dfs_total / n_state`. Zero for a pure
least-squares fit, one when the measurement carries no information.
"""
prior_influence(dfs_total::Real, n_state::Integer) = 1 - Float64(dfs_total) / n_state

"""
    normal_matrix_condition(N) -> Float64
    normal_matrix_condition(K, obs_precision, prior_precision) -> Float64

Condition number (2-norm) of the normal matrix after symmetric diagonal (Jacobi) scaling,
`D^(-1/2) * N * D^(-1/2)` with `D = diag(N)`. The scaling makes the number independent of the
state units (a column in molecules/cm2 next to a dimensionless baseline coefficient), so it
measures how well the measurement and prior separate the state elements. Returns `Inf` when a
diagonal entry is zero or the matrix is not positive definite.

```julia
normal_matrix_condition(K, Se_inv, Sa_inv) < 1e12
```
"""
function normal_matrix_condition(N::AbstractMatrix)
    d = diag(N)
    (all(isfinite, N) && all(>(0), d)) || return Inf
    s = 1 ./ sqrt.(d)
    M = Symmetric(Matrix{Float64}(s .* N .* s'))
    lambda = eigvals(M)
    lambda_min = minimum(lambda)
    return lambda_min > 0 ? maximum(lambda) / lambda_min : Inf
end
normal_matrix_condition(K::AbstractMatrix, obs_precision::ObservationPrecision,
                        prior_precision::AbstractMatrix) =
    normal_matrix_condition(normal_matrix(K, obs_precision, prior_precision))

"""
    predicted_cost(y, y_model, K, x, x_prior, obs_precision, prior_precision, dx) -> Float64

Cost predicted by the linearization at the pre-step state `x` (where the model is `y_model`
and the Jacobian is `K`) for the step `dx`:
`(y - y_model - K dx)' * obs_precision * (...) + (x + dx - x_prior)' * prior_precision * (...)`.
Pure: it uses only pre-step quantities (issue 6). On a linear model it equals the true cost
after the step.
"""
function predicted_cost(y::AbstractVector, y_model::AbstractVector, K::AbstractMatrix,
                        x::AbstractVector, x_prior::AbstractVector,
                        obs_precision::ObservationPrecision, prior_precision::AbstractMatrix,
                        dx::AbstractVector)
    r = y - y_model - K * dx
    p = collect(x) + dx - collect(x_prior)
    return Float64(dot(r, obs_precision * r) + dot(p, prior_precision * p))
end

"""
    gain_ratio(cost_before, cost_after, cost_predicted) -> Float64

Ratio of actual to predicted cost reduction,
`(cost_before - cost_after) / (cost_before - cost_predicted)`. Exactly 1 on a linear problem;
adaptive damping lowers the damping when it is large and raises it when it is small.
"""
gain_ratio(cost_before::Real, cost_after::Real, cost_predicted::Real) =
    Float64((cost_before - cost_after) / (cost_before - cost_predicted))

# ---------------------------------------------------------------- report text

_fmt(v::Real) = isfinite(v) ? @sprintf("%.6g", v) : string(v)

function _csv_field(name::AbstractString)
    needs_quotes = any(c -> c in (',', '"', '\n'), name)
    return needs_quotes ? string('"', replace(name, "\"" => "\"\""), '"') : String(name)
end

# Builds the report text for `report(result::RetrievalResult; state_names)`, defined in
# retrieval_result.jl after the result type exists. ASCII only.
function _report_text(result, state_names::AbstractVector{<:AbstractString})
    n_state = length(result.x)
    length(state_names) == n_state ||
        throw(ArgumentError("state_names: has $(length(state_names)) names, the state has $n_state elements"))
    sigma = sqrt.(max.(diag(result.posterior_covariance), 0.0))
    ak = diag(result.averaging_kernel)
    dfs = collect(result.dfs_per_element)
    io = IOBuffer()
    println(io, "Optimal estimation report")
    println(io, "status: ", result.status)
    isempty(result.message) || println(io, "message: ", result.message)
    println(io, "iterations: ", result.n_iterations)
    println(io, "n_obs: ", length(result.y), ", n_state: ", n_state)
    println(io, "DFS total: ", _fmt(result.dfs_total))
    println(io, "chi2 obs per dof: ", _fmt(result.chi2_obs_per_dof))
    println(io, "chi2 total per dof: ", _fmt(result.chi2_total_per_dof))
    println(io, "residual RMSE (measurement units, full-weight observations): ", _fmt(result.rmse))
    println(io, "Rodgers d2 per state: ", _fmt(result.rodgers_d2_per_state))
    println(io, "prior influence: ", _fmt(result.prior_influence))
    println(io, "condition number (scaled normal matrix): ", _fmt(result.condition_number))
    println(io)
    println(io, "```csv")
    println(io, "name,prior,retrieved,posterior_sigma,dfs,ak_diagonal")
    for i in 1:n_state
        println(io, join((_csv_field(state_names[i]), _fmt(result.x_prior[i]), _fmt(result.x[i]),
                          _fmt(sigma[i]), _fmt(dfs[i]), _fmt(ak[i])), ","))
    end
    println(io, "```")
    text = String(take!(io))
    # names come from the caller; replace any non-ASCII character so the report stays ASCII
    return all(isascii, text) ? text : map(c -> isascii(c) ? c : '?', text)
end

export normal_matrix, posterior_covariance, scaled_inverse, gain_matrix, averaging_kernel
"""
    column_averaging_kernel(averaging_kernel, indices, weights) -> Vector{Float64}

Column averaging kernel of the profile at state `indices`: `(h' A) ./ h'` with
`h = weights / sum(weights)` and `A` the block of the averaging kernel for those indices.
With layer dry-air columns as `weights` and a state in layer columns it is the sensitivity
of the column-mean mole fraction to each layer (1 for a perfect retrieval).

```julia
i = state_indices(result.x, :CH4)
column_averaging_kernel(result.averaging_kernel, i, layer_dry_air_columns)
```
"""
function column_averaging_kernel(A::AbstractMatrix, indices::AbstractVector{<:Integer},
                                 weights::AbstractVector{<:Real})
    length(indices) == length(weights) || throw(DimensionMismatch(
        "column_averaging_kernel: $(length(indices)) state indices, $(length(weights)) weights"))
    h = weights ./ sum(weights)
    return vec((h' * A[indices, indices]) ./ h')
end

export dfs_per_element, dfs_total, rodgers_d2, column_averaging_kernel
export observation_cost, prior_cost, total_cost, chi2_per_dof, prior_influence
export residual_rmse, full_weight_observations
export normal_matrix_condition, predicted_cost, gain_ratio
