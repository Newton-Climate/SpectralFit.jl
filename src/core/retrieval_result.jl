# Retrieval result and the one shared constructor that computes every diagnostic
# (section 3.11 of docs/architecture_plan.md). Solvers only return the final state, Jacobian,
# model and iteration history; the posterior is built here and nowhere else (issue 1).

"""
    RetrievalStatus

Outcome of `solve`: `Converged` (Rodgers d2 per state fell below the tolerance on an accepted
step), `MaxIterations` (the iteration limit was reached first; never reported as converged),
or `Failed` (a numerical failure such as a singular normal matrix or non-finite model output;
`result.message` says which).
"""
@enum RetrievalStatus Converged MaxIterations Failed

@doc "Status of a retrieval whose Rodgers test d2 / n_state fell below the tolerance." Converged
@doc "Status of a retrieval that reached `max_iterations` without meeting the Rodgers test." MaxIterations
@doc "Status of a retrieval that stopped on a numerical failure; `result.message` says which." Failed

"""
    RetrievalResult

Outcome of a retrieval with all optimal-estimation diagnostics. Fields:

- `x`, `x_prior`: retrieved and prior state (`ComponentVector`, state units).
- `status::RetrievalStatus`, `message::String`, `n_iterations::Int`.
- `rodgers_d2_per_state`: final value of the convergence statistic `d2 / n_state`.
- `cost_history`: total cost at the prior and after each accepted step.
- `y`, `y_model`: measurement and model at `x`, in measurement space.
- `K`: Jacobian at `x` (n_obs x n_state).
- `obs_precision`, `prior_precision`: the problem's own precisions (issue 7); `obs_precision`
  may be a `CovariancePrecision`.
- `posterior_covariance`: `inv(K' obs_precision K + prior_precision)`, state units squared.
- `averaging_kernel`: `posterior_covariance * K' * obs_precision * K`.
- `dfs_per_element`: diagonal of the averaging kernel, with the axes of `x` (so
  `result.dfs_per_element.CH4` works); `dfs_total` is its sum.
- `chi2_obs_per_dof`: observation cost over `n_obs - n_state`.
- `chi2_total_per_dof`: full cost (observation plus prior) over `n_obs - n_state`.
- `prior_influence`: `1 - dfs_total / n_state`.
- `condition_number`: condition number of the Jacobi-scaled normal matrix at `x`
  (see `normal_matrix_condition`).
- `rmse`: root-mean-square residual `y - y_model` in measurement-space units (`residual_rmse`),
  over the observations at full weight (de-weighted, masked observations are left out).

For a `Failed` result the matrix diagnostics that could not be computed are `NaN`.
Built only by `retrieval_result`.
"""
struct RetrievalResult{X<:ComponentVector, PO<:ObservationPrecision, PA<:AbstractMatrix}
    x::X
    x_prior::X
    status::RetrievalStatus
    message::String
    n_iterations::Int
    rodgers_d2_per_state::Float64
    cost_history::Vector{Float64}
    y::Vector{Float64}
    y_model::Vector{Float64}
    K::Matrix{Float64}
    obs_precision::PO
    prior_precision::PA
    posterior_covariance::Matrix{Float64}
    averaging_kernel::Matrix{Float64}
    dfs_per_element::X
    dfs_total::Float64
    chi2_obs_per_dof::Float64
    chi2_total_per_dof::Float64
    prior_influence::Float64
    condition_number::Float64
    rmse::Float64
end

"""
    retrieval_result(problem, x, K, y_model; status, message = "", n_iterations,
                     rodgers_d2_per_state, cost_history) -> RetrievalResult

The single constructor of `RetrievalResult`, used by every solver. `problem` is a
`RetrievalProblem`; `x` is the final state (flat vector or `ComponentVector` in state units);
`K` and `y_model` are the Jacobian and model at `x`. Computes the posterior covariance,
averaging kernel, DFS, chi squared and prior influence from `core/diagnostics.jl`.

If the model or Jacobian is not finite or the normal matrix is not positive definite, the
matrix diagnostics are `NaN` and the status becomes `Failed` (the message says why).

```julia
retrieval_result(problem, x, K, y_model; status = Converged, n_iterations = 3,
                 rodgers_d2_per_state = 1e-4, cost_history = [120.0, 21.0, 20.1, 20.1])
```
"""
function retrieval_result(problem, x::AbstractVector, K::AbstractMatrix, y_model::AbstractVector;
                          status::RetrievalStatus, message::AbstractString = "",
                          n_iterations::Integer, rodgers_d2_per_state::Real,
                          cost_history::AbstractVector{<:Real})
    x_prior = problem.x_prior
    axes_x = getaxes(x_prior)
    x_flat = Vector{Float64}(collect(x))
    x_cv = ComponentVector(x_flat, axes_x)
    y = problem.y
    Se_inv = problem.obs_precision
    Sa_inv = problem.prior_precision
    n_obs, n_state = length(y), length(x_flat)
    K_dense = Matrix{Float64}(K)
    y_model_vec = Vector{Float64}(collect(y_model))
    msg = String(message)

    S = fill(NaN, n_state, n_state)
    A = fill(NaN, n_state, n_state)
    condition = Inf
    if !(all(isfinite, K_dense) && all(isfinite, y_model_vec))
        status = Failed
        msg = _append_message(msg, "model output or Jacobian is not finite at the final state")
    else
        N = normal_matrix(K_dense, Se_inv, Sa_inv)
        if issuccess(cholesky(N; check = false))
            S = scaled_inverse(N)
            A = S * Matrix{Float64}(K_dense' * (Se_inv * K_dense))
            condition = normal_matrix_condition(N)
        else
            status = Failed
            msg = _append_message(msg, "normal matrix K'*obs_precision*K + prior_precision " *
                                       "is not positive definite at the final state")
        end
    end

    dfs = dfs_per_element(A)
    dfs_sum = sum(dfs)
    cost_obs = observation_cost(y, y_model_vec, Se_inv)
    cost_prior = prior_cost(x_flat, x_prior, Sa_inv)
    return RetrievalResult(x_cv, x_prior, status, msg, Int(n_iterations),
                           Float64(rodgers_d2_per_state), Vector{Float64}(cost_history),
                           copy(y), y_model_vec, K_dense, Se_inv, Sa_inv, S, A,
                           ComponentVector(dfs, axes_x), dfs_sum,
                           chi2_per_dof(cost_obs, n_obs, n_state),
                           chi2_per_dof(cost_obs + cost_prior, n_obs, n_state),
                           prior_influence(dfs_sum, n_state), condition,
                           residual_rmse(y, y_model_vec, Se_inv))
end

_append_message(msg::String, extra::String) = isempty(msg) ? extra : string(msg, "; ", extra)

"""
    state_labels(x::ComponentVector) -> Vector{String}

One plain-text name per state element from the `ComponentVector` labels, for example
`["CH4", "baseline.window_1[1]", ...]`.
"""
state_labels(x::ComponentVector) = Vector{String}(labels(x))

"""
    report(result::RetrievalResult; state_names = state_labels(result.x)) -> String

Plain-text optimal-estimation report (ASCII only, no Greek letters): status, message,
iterations, DFS total, chi2 per dof (observation and total), Rodgers d2 per state, prior
influence and condition number, followed by a fenced CSV block with one row per state element
and the columns `name,prior,retrieved,posterior_sigma,dfs,ak_diagonal` (prior, retrieved and
posterior sigma in state units). Printing it is the caller's choice.

```julia
result = solve(problem, GaussNewton())
text = report(result)
```
"""
report(result::RetrievalResult; state_names::AbstractVector{<:AbstractString} = state_labels(result.x)) =
    _report_text(result, state_names)

Base.show(io::IO, ::MIME"text/plain", result::RetrievalResult) = print(io, report(result))

export RetrievalStatus, Converged, MaxIterations, Failed
export RetrievalResult, retrieval_result, state_labels
