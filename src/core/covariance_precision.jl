# An observation precision held as the sparse Cholesky factor of its covariance
# (section 8, open question 9 of docs/architecture_plan.md). The inverse of a banded covariance
# is dense, so the precision is never formed: every product `precision * v` is a solve with the
# factor. `RetrievalProblem`, the solvers, the diagnostics and `retrieval_result` only ever
# multiply the observation precision by a vector or matrix from the left, which is all this
# type provides.

"""
    CovariancePrecision(covariance)

Observation precision `Se_inv = inv(covariance)` represented by the sparse Cholesky
factorization of `covariance` (n_obs x n_obs, symmetric positive definite, measurement units
squared), for example the banded matrix from `wms_noise_covariance`. Pass it wherever an
`obs_precision` is expected:

```julia
Se = wms_noise_covariance(prepared, lockin, detector_variance; signals = signals)
problem = RetrievalProblem(model, y, CovariancePrecision(Se), x_prior, Sa_inv)
```

`P * v` and `P * M` solve with the factor (no dense inverse is formed); `size(P)`,
`observation_covariance(P)` and `Matrix(P)` (the dense inverse, for small problems and tests)
are also defined. It is not an `AbstractMatrix`, so indexing is not supported. Throws an
`ArgumentError` naming `covariance` if it is not square, finite, symmetric (to 1e-10
relative) and positive definite.
"""
struct CovariancePrecision{C<:SparseMatrixCSC{Float64}, F}
    covariance::C
    factor::F
    function CovariancePrecision(covariance::AbstractMatrix{<:Real})
        n, m = size(covariance)
        n == m || throw(ArgumentError("covariance: must be square, got size $((n, m))"))
        S = SparseMatrixCSC{Float64,Int}(sparse(covariance))
        all(isfinite, nonzeros(S)) || throw(ArgumentError("covariance: contains non-finite values"))
        scale = maximum(abs, nonzeros(S); init = 0.0)
        asymmetry = maximum(abs, nonzeros(S - S'); init = 0.0)
        (scale > 0 && asymmetry <= 1e-10 * scale) ||
            throw(ArgumentError("covariance: is not symmetric (or is zero)"))
        S = (S + S') ./ 2
        factor = cholesky(Symmetric(S); check = false)
        issuccess(factor) || throw(ArgumentError(
            "covariance: is not positive definite (sparse Cholesky failed); for a truncated " *
            "band, lower band_tolerance"))
        return new{typeof(S), typeof(factor)}(S, factor)
    end
end

Base.size(P::CovariancePrecision) = size(P.covariance)
Base.size(P::CovariancePrecision, d::Integer) = size(P.covariance, d)
Base.:*(P::CovariancePrecision, v::AbstractVector{<:Real}) = P.factor \ Vector{Float64}(v)
Base.:*(P::CovariancePrecision, M::AbstractMatrix{<:Real}) = P.factor \ Matrix{Float64}(M)
Base.Matrix(P::CovariancePrecision) = Matrix{Float64}(inv(Symmetric(Matrix(P.covariance))))

"""`observation_covariance(P::CovariancePrecision)`: the sparse covariance `inv(P)`."""
observation_covariance(P::CovariancePrecision) = P.covariance

# `A' * P` for the gain matrix; P is symmetric, so it is `(P * A)'`. Dense and Diagonal
# precisions keep the plain product.
_adjoint_times_precision(A::AbstractMatrix, P::AbstractMatrix) = A' * P
_adjoint_times_precision(A::AbstractMatrix, P::CovariancePrecision) = Matrix((P * A)')

export CovariancePrecision, observation_covariance
