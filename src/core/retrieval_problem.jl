# Retrieval problem: forward model, measurement, precisions and prior, validated at
# construction (section 3.11 of docs/architecture_plan.md), plus the Jacobian machinery
# (ForwardDiff configuration and result buffer built once per problem).

# The forward model takes a ComponentVector; ForwardDiff works on a flat Vector. This wrapper
# rebuilds the ComponentVector (of Float64 or dual numbers) with the prior's axes.
struct FlatForwardModel{F, A}
    forward_model::F
    state_axes::A
end
(m::FlatForwardModel)(v::AbstractVector) = m.forward_model(ComponentVector(v, m.state_axes))

# ForwardDiff configuration and DiffResults buffer, allocated once per problem. They are
# mutated by every Jacobian evaluation, so one problem must not be solved from several
# threads at once (build one problem per measurement, as the batch app does).
struct JacobianWorkspace{M<:FlatForwardModel, C, R}
    flat_model::M
    config::C
    result::R
end

"""
    RetrievalProblem(forward_model, y, obs_precision, x_prior, prior_precision)

A retrieval: find the state `x` that minimizes
`(y - F(x))' * obs_precision * (y - F(x)) + (x - x_prior)' * prior_precision * (x - x_prior)`.

- `forward_model`: any callable `F(x::ComponentVector) -> AbstractVector` in measurement space,
  generic in element type so ForwardDiff dual numbers pass through.
- `y`: measurement in measurement space (length `n_obs`).
- `obs_precision`: inverse observation covariance Se_inv (`n_obs x n_obs`), symmetric positive
  definite, e.g. `Diagonal(1 ./ sigma_y .^ 2)`.
- `x_prior`: prior and first-guess state, a `ComponentVector` (converted to Float64).
- `prior_precision`: inverse prior covariance Sa_inv (`n_state x n_state`), symmetric positive
  semi-definite. A zero matrix gives an unconstrained least-squares fit.

Validated at construction, with an `ArgumentError` naming the argument: dimensions (`y`
against `F(x_prior)`, both precisions against `n_obs` and `n_state`), finite values,
symmetry, positive definiteness of `obs_precision`, positive semi-definiteness of
`prior_precision`, and the condition number of the Jacobi-scaled normal matrix
`K' * obs_precision * K + prior_precision` at `x_prior`, rejected above 1e12 (see
`normal_matrix_condition`).

```julia
x_prior = ComponentVector(a = 1.0, b = [0.0, 0.0])
model(x) = [x.a + x.b[1], x.a - x.b[2], x.b[1] * x.b[2], x.a]
problem = RetrievalProblem(model, [1.2, 0.9, 0.0, 1.1], Diagonal(fill(1e4, 4)),
                           x_prior, Diagonal([1.0, 1.0, 1.0]))
```
"""
struct RetrievalProblem{F, X<:ComponentVector, PO<:AbstractMatrix, PA<:AbstractMatrix,
                        W<:JacobianWorkspace}
    forward_model::F
    y::Vector{Float64}
    obs_precision::PO
    x_prior::X
    prior_precision::PA
    jacobian_workspace::W

    function RetrievalProblem(forward_model, y::AbstractVector{<:Real},
                              obs_precision::AbstractMatrix{<:Real},
                              x_prior::ComponentVector,
                              prior_precision::AbstractMatrix{<:Real})
        x0 = ComponentVector(copyto!(Vector{Float64}(undef, length(x_prior)), x_prior), getaxes(x_prior))
        y0 = collect(Float64, y)
        n_state = length(x0)
        n_obs = length(y0)
        n_state > 0 || throw(ArgumentError("x_prior: the state vector is empty"))
        all(isfinite, x0) || throw(ArgumentError("x_prior: contains non-finite values"))
        all(isfinite, y0) || throw(ArgumentError("y: contains non-finite values"))
        n_model = length(forward_model(x0))
        n_model == n_obs || throw(ArgumentError(
            "y: has length $n_obs but forward_model(x_prior) has length $n_model"))
        _check_precision(obs_precision, n_obs, "obs_precision", "n_obs"; semidefinite = false)
        _check_precision(prior_precision, n_state, "prior_precision", "n_state"; semidefinite = true)

        workspace = _jacobian_workspace(forward_model, x0, n_obs)
        v0 = collect(getdata(x0))
        F0, K0 = _model_and_jacobian!(workspace, v0)
        (all(isfinite, F0) && all(isfinite, K0)) || throw(ArgumentError(
            "forward_model: output or Jacobian at x_prior is not finite"))
        condition = normal_matrix_condition(normal_matrix(K0, obs_precision, prior_precision))
        condition <= 1e12 || throw(ArgumentError(
            "prior_precision: condition number $(condition) of the scaled normal matrix " *
            "K'*obs_precision*K + prior_precision at x_prior exceeds 1e12; the forward_model " *
            "does not constrain every state element and prior_precision does not make up for it"))
        return new{typeof(forward_model), typeof(x0), typeof(obs_precision),
                   typeof(prior_precision), typeof(workspace)}(
            forward_model, y0, obs_precision, x0, prior_precision, workspace)
    end
end

function _jacobian_workspace(forward_model, x0::ComponentVector, n_obs::Integer)
    flat = FlatForwardModel(forward_model, getaxes(x0))
    v0 = collect(getdata(x0))
    config = ForwardDiff.JacobianConfig(flat, v0)
    result = DiffResults.JacobianResult(zeros(n_obs), v0)
    return JacobianWorkspace(flat, config, result)
end

function _model_and_jacobian!(w::JacobianWorkspace, v::AbstractVector{Float64})
    ForwardDiff.jacobian!(w.result, w.flat_model, v, w.config)
    y_model = Vector{Float64}(DiffResults.value(w.result))
    K = Matrix{Float64}(DiffResults.jacobian(w.result))
    return y_model, K
end

function _check_precision(P::AbstractMatrix, n::Integer, name::String, size_name::String;
                          semidefinite::Bool)
    size(P) == (n, n) || throw(ArgumentError(
        "$name: has size $(size(P)) but must be ($n, $n) to match $size_name = $n"))
    all(isfinite, P) || throw(ArgumentError("$name: contains non-finite values"))
    _is_symmetric(P) || throw(ArgumentError("$name: is not symmetric"))
    if semidefinite
        _is_positive_semidefinite(P) || throw(ArgumentError(
            "$name: is not positive semi-definite (an eigenvalue is negative)"))
    else
        _is_positive_definite(P) || throw(ArgumentError("$name: is not positive definite"))
    end
    return nothing
end

_is_symmetric(P::Diagonal) = true
function _is_symmetric(P::AbstractMatrix)
    scale = maximum(abs, P; init = 0.0)
    return scale == 0 || maximum(abs, P - P'; init = 0.0) <= 1e-10 * scale
end

_is_positive_definite(P::Diagonal) = all(>(0), P.diag)
_is_positive_definite(P::AbstractMatrix) =
    issuccess(cholesky(Symmetric((P + P') / 2); check = false))

_is_positive_semidefinite(P::Diagonal) = all(>=(0), P.diag)
function _is_positive_semidefinite(P::AbstractMatrix)
    M = Symmetric(Matrix{Float64}((P + P') / 2))
    scale = maximum(abs, M; init = 0.0)
    return scale == 0 || eigmin(M) >= -1e-10 * scale
end

"""
    model_and_jacobian(problem, x) -> (y_model::Vector{Float64}, K::Matrix{Float64})

Forward model and its ForwardDiff Jacobian (n_obs x n_state, measurement units per state
unit) at state `x` (flat vector or `ComponentVector` with the prior's layout), using the
problem's preallocated configuration and buffer.

```julia
y_model, K = model_and_jacobian(problem, problem.x_prior)
```
"""
model_and_jacobian(problem::RetrievalProblem, x::AbstractVector) =
    _model_and_jacobian!(problem.jacobian_workspace, collect(Float64, x))

"""
    evaluate_model(problem, x) -> Vector{Float64}

Forward model at state `x` (flat vector or `ComponentVector`), without the Jacobian.
"""
evaluate_model(problem::RetrievalProblem, x::AbstractVector) =
    Vector{Float64}(problem.jacobian_workspace.flat_model(collect(Float64, x)))

"""`n_state(problem)`: number of state elements."""
n_state(problem::RetrievalProblem) = length(problem.x_prior)
"""`n_obs(problem)`: number of observations."""
n_obs(problem::RetrievalProblem) = length(problem.y)

export RetrievalProblem, model_and_jacobian, evaluate_model, n_state, n_obs
