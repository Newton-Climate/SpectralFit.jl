# Solvers: Gauss-Newton (maximum a posteriori form) and Levenberg-Marquardt with fixed or
# adaptive damping (section 3.11 of docs/architecture_plan.md). One iteration loop serves all
# of them; each solver only proposes the step. Diagnostics come from `retrieval_result`.

"""
    GaussNewton(; max_iterations = 30, rodgers_tolerance = 0.01)

Gauss-Newton iteration in the maximum a posteriori form (Rodgers eq. 5.9):
`x_next = x + (K' Se_inv K + Sa_inv) \\ (K' Se_inv (y - F(x)) - Sa_inv (x - x_prior))`,
starting at `x_prior`. Converged when `rodgers_d2 / n_state < rodgers_tolerance` (both
dimensionless).

```julia
result = solve(problem, GaussNewton(max_iterations = 10))
```
"""
struct GaussNewton <: AbstractSolver
    max_iterations::Int
    rodgers_tolerance::Float64
    function GaussNewton(; max_iterations::Integer = 30, rodgers_tolerance::Real = 0.01)
        _check_iteration_settings(max_iterations, rodgers_tolerance)
        return new(max_iterations, rodgers_tolerance)
    end
end

"""
    FixedDamping(gamma)

Constant Levenberg-Marquardt damping `gamma >= 0` (dimensionless). With prior scaling the
step matrix is `K' Se_inv K + (1 + gamma) Sa_inv` (Rodgers eq. 5.36). Every step is accepted.

```julia
LevenbergMarquardt(damping = FixedDamping(1.0))
```
"""
struct FixedDamping
    gamma::Float64
    function FixedDamping(gamma::Real)
        (isfinite(gamma) && gamma >= 0) ||
            throw(ArgumentError("gamma: must be finite and non-negative, got $gamma"))
        return new(gamma)
    end
end

"""
    AdaptiveDamping(; initial = 1.0, lower_ratio = 0.25, upper_ratio = 0.75, factor = 2.0,
                    max_retries = 30)

Adaptive Levenberg-Marquardt damping. A trial step that increases the cost is rejected, the
damping is multiplied by `factor` and the step is retried (at most `max_retries` times per
iteration, after which the solve fails). On an accepted step the gain ratio, actual over
predicted cost reduction with `predicted_cost` evaluated at the pre-step state, sets the next
damping: divided by `factor` above `upper_ratio`, multiplied by `factor` below `lower_ratio`.

```julia
LevenbergMarquardt(damping = AdaptiveDamping(initial = 10.0), scaling = :marquardt)
```
"""
struct AdaptiveDamping
    initial::Float64
    lower_ratio::Float64
    upper_ratio::Float64
    factor::Float64
    max_retries::Int
    function AdaptiveDamping(; initial::Real = 1.0, lower_ratio::Real = 0.25,
                             upper_ratio::Real = 0.75, factor::Real = 2.0,
                             max_retries::Integer = 30)
        (isfinite(initial) && initial >= 0) ||
            throw(ArgumentError("initial: must be finite and non-negative, got $initial"))
        0 <= lower_ratio <= upper_ratio ||
            throw(ArgumentError("lower_ratio, upper_ratio: need 0 <= lower_ratio <= upper_ratio"))
        factor > 1 || throw(ArgumentError("factor: must be greater than 1, got $factor"))
        max_retries >= 0 || throw(ArgumentError("max_retries: must be non-negative"))
        return new(initial, lower_ratio, upper_ratio, factor, max_retries)
    end
end

"""
    LevenbergMarquardt(; damping = AdaptiveDamping(), scaling = :prior, max_iterations = 30,
                       rodgers_tolerance = 0.01)

Levenberg-Marquardt iteration: the Gauss-Newton step with the normal matrix replaced by
`K' Se_inv K + Sa_inv + gamma * D`. `scaling = :prior` uses `D = Sa_inv`; `scaling =
:marquardt` uses `D = Diagonal(diag(K' Se_inv K))`. With prior scaling, state elements whose
prior precision diagonal is zero (for example a zero prior, the least-squares mode) get the
Marquardt diagonal instead, so every element is damped. `damping` is `FixedDamping(gamma)`
or `AdaptiveDamping(...)`. Convergence uses the undamped normal matrix in the Rodgers test on
accepted steps.

```julia
solve(problem, LevenbergMarquardt(damping = FixedDamping(1.0), max_iterations = 100))
```
"""
struct LevenbergMarquardt{D<:Union{FixedDamping, AdaptiveDamping}} <: AbstractSolver
    damping::D
    scaling::Symbol
    max_iterations::Int
    rodgers_tolerance::Float64
    function LevenbergMarquardt(; damping::Union{FixedDamping, AdaptiveDamping} = AdaptiveDamping(),
                                scaling::Symbol = :prior, max_iterations::Integer = 30,
                                rodgers_tolerance::Real = 0.01)
        scaling in (:prior, :marquardt) ||
            throw(ArgumentError("scaling: must be :prior or :marquardt, got :$scaling"))
        _check_iteration_settings(max_iterations, rodgers_tolerance)
        return new{typeof(damping)}(damping, scaling, max_iterations, rodgers_tolerance)
    end
end

function _check_iteration_settings(max_iterations, rodgers_tolerance)
    max_iterations >= 1 || throw(ArgumentError("max_iterations: must be at least 1"))
    (isfinite(rodgers_tolerance) && rodgers_tolerance > 0) ||
        throw(ArgumentError("rodgers_tolerance: must be positive and finite"))
    return nothing
end

"""
    solve(problem::RetrievalProblem, solver::GaussNewton) -> RetrievalResult
    solve(problem::RetrievalProblem, solver::LevenbergMarquardt) -> RetrievalResult

Iterate from `problem.x_prior` until the Rodgers test passes on an accepted step (status
`Converged`) or `solver.max_iterations` iterations are done (`MaxIterations`). A singular
normal matrix, non-finite model output, or an adaptive-damping search that cannot reduce the
cost returns a result with status `Failed` and an explanatory `message`. Exceptions thrown by
the forward model itself propagate. Does not print.

```julia
result = solve(problem, GaussNewton())
result.status == Converged
result.dfs_per_element.CH4
```
"""
solve(problem::RetrievalProblem, solver::GaussNewton) =
    _solve(problem, solver, solver.max_iterations, solver.rodgers_tolerance)
solve(problem::RetrievalProblem, solver::LevenbergMarquardt) =
    _solve(problem, solver, solver.max_iterations, solver.rodgers_tolerance)

_initial_damping(::GaussNewton) = 0.0
_initial_damping(s::LevenbergMarquardt{FixedDamping}) = s.damping.gamma
_initial_damping(s::LevenbergMarquardt{AdaptiveDamping}) = s.damping.initial

_finite(F, K) = all(isfinite, F) && all(isfinite, K)

function _solve(problem::RetrievalProblem, solver::AbstractSolver, max_iterations::Int,
                tolerance::Float64)
    x_prior = collect(getdata(problem.x_prior))
    y, Se_inv, Sa_inv = problem.y, problem.obs_precision, problem.prior_precision
    n = length(x_prior)
    v = copy(x_prior)
    F, K = model_and_jacobian(problem, v)
    cost = total_cost(y, F, Se_inv, v, x_prior, Sa_inv)
    history = [cost]
    if !_finite(F, K)
        return retrieval_result(problem, v, K, F; status = Failed, n_iterations = 0,
                                message = "forward_model output or Jacobian is not finite at x_prior",
                                rodgers_d2_per_state = NaN, cost_history = history)
    end
    damping = _initial_damping(solver)
    d2_per_state = NaN
    n_rejected = 0
    for iteration in 1:max_iterations
        information = Matrix{Float64}(K' * (Se_inv * K))
        N0 = information + Sa_inv
        N = Symmetric((N0 + N0') ./ 2)
        g = K' * (Se_inv * (y - F)) - Sa_inv * (v - x_prior)
        step = _propose_step(solver, problem, v, x_prior, F, K, information, N, g, cost,
                             damping, tolerance)
        n_rejected += step.n_rejected
        damping = step.damping
        if !step.ok
            return retrieval_result(problem, v, K, F; status = Failed,
                                    message = _with_rejections(step.message, n_rejected),
                                    n_iterations = iteration - 1,
                                    rodgers_d2_per_state = d2_per_state, cost_history = history)
        end
        v = v + step.dx
        d2_per_state = step.d2 / n
        F, K = model_and_jacobian(problem, v)
        if !_finite(F, K)
            return retrieval_result(problem, v, K, F; status = Failed, n_iterations = iteration,
                                    message = "forward_model output or Jacobian is not finite " *
                                              "after iteration $iteration",
                                    rodgers_d2_per_state = d2_per_state, cost_history = history)
        end
        cost = total_cost(y, F, Se_inv, v, x_prior, Sa_inv)
        push!(history, cost)
        if d2_per_state < tolerance
            msg = @sprintf("Rodgers d2 per state %.3g below tolerance %.3g after %d iterations",
                           d2_per_state, tolerance, iteration)
            return retrieval_result(problem, v, K, F; status = Converged,
                                    message = _with_rejections(msg, n_rejected),
                                    n_iterations = iteration,
                                    rodgers_d2_per_state = d2_per_state, cost_history = history)
        end
    end
    msg = @sprintf("reached max_iterations = %d with Rodgers d2 per state %.3g above tolerance %.3g",
                   max_iterations, d2_per_state, tolerance)
    return retrieval_result(problem, v, K, F; status = MaxIterations,
                            message = _with_rejections(msg, n_rejected),
                            n_iterations = max_iterations,
                            rodgers_d2_per_state = d2_per_state, cost_history = history)
end

_with_rejections(msg::String, n_rejected::Int) =
    n_rejected == 0 ? msg : string(msg, "; adaptive damping rejected ", n_rejected, " steps")

const _SINGULAR_MESSAGE = "normal matrix K'*obs_precision*K + prior_precision is singular " *
                          "or not positive definite"

_step_result(ok, dx, d2, damping, n_rejected, message) =
    (ok = ok, dx = dx, d2 = Float64(d2), damping = Float64(damping),
     n_rejected = n_rejected, message = message)

function _propose_step(::GaussNewton, problem, v, x_prior, F, K, information, N, g, cost,
                       damping, tolerance)
    C = cholesky(N; check = false)
    issuccess(C) || return _step_result(false, zero(g), NaN, damping, 0, _SINGULAR_MESSAGE)
    dx = C \ g
    return _step_result(true, dx, rodgers_d2(dx, N), damping, 0, "")
end

# Damping matrix D of Levenberg-Marquardt as a dense matrix (see the solver docstring).
function _damping_matrix(scaling::Symbol, information::Matrix{Float64}, Sa_inv::AbstractMatrix)
    marquardt = diag(information)
    scaling === :marquardt && return Matrix{Float64}(Diagonal(marquardt))
    D = Matrix{Float64}(Sa_inv)
    for i in axes(D, 1)
        D[i, i] == 0 && (D[i, i] = marquardt[i])
    end
    return D
end

function _propose_step(solver::LevenbergMarquardt{FixedDamping}, problem, v, x_prior, F, K,
                       information, N, g, cost, damping, tolerance)
    D = _damping_matrix(solver.scaling, information, problem.prior_precision)
    C = cholesky(Symmetric(Matrix(N) + damping * D); check = false)
    issuccess(C) || return _step_result(false, zero(g), NaN, damping, 0, _SINGULAR_MESSAGE)
    dx = C \ g
    return _step_result(true, dx, rodgers_d2(dx, N), damping, 0, "")
end

function _propose_step(solver::LevenbergMarquardt{AdaptiveDamping}, problem, v, x_prior, F, K,
                       information, N, g, cost, damping, tolerance)
    settings = solver.damping
    y, Se_inv, Sa_inv = problem.y, problem.obs_precision, problem.prior_precision
    D = _damping_matrix(solver.scaling, information, Sa_inv)
    N_dense = Matrix(N)
    gamma = damping
    n_rejected = 0
    for _ in 0:settings.max_retries
        C = cholesky(Symmetric(N_dense + gamma * D); check = false)
        if issuccess(C)
            dx = C \ g
            d2 = rodgers_d2(dx, N)
            v_trial = v + dx
            F_trial = evaluate_model(problem, v_trial)
            cost_trial = all(isfinite, F_trial) ?
                         total_cost(y, F_trial, Se_inv, v_trial, x_prior, Sa_inv) : Inf
            # a step below the convergence threshold is accepted even if rounding raises the cost
            if isfinite(cost_trial) && (cost_trial <= cost || d2 / length(v) < tolerance)
                cost_pred = predicted_cost(y, F, K, v, x_prior, Se_inv, Sa_inv, dx)
                rho = gain_ratio(cost, cost_trial, cost_pred)
                if rho > settings.upper_ratio
                    gamma /= settings.factor
                elseif rho < settings.lower_ratio
                    gamma *= settings.factor
                end
                return _step_result(true, dx, d2, gamma, n_rejected, "")
            end
        end
        n_rejected += 1
        gamma = gamma == 0 ? 1.0 : gamma * settings.factor
    end
    return _step_result(false, zero(g), NaN, gamma, n_rejected,
                        "adaptive damping found no cost-reducing step after " *
                        "$(settings.max_retries) retries")
end

export GaussNewton, LevenbergMarquardt, FixedDamping, AdaptiveDamping
