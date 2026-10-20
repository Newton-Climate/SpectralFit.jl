# Tests for src/core/solvers.jl on problems with known answers.

# Linear-Gaussian problem with random K, Se and Sa from a fixed seed.
function phase5_linear_problem(; seed = 42, n_obs = 30, prior = :informative)
    rng = MersenneTwister(seed)
    x_prior = ComponentVector(a = 1.0, b = [0.5, -0.3])
    K = randn(rng, n_obs, 3)
    noise_sigma = 0.05
    Se_inv = Diagonal(fill(1 / noise_sigma^2, n_obs))
    Sa_inv = prior === :zero ? zeros(3, 3) :
             prior === :weak ? Matrix(1e-8I, 3, 3) :
             (B = randn(rng, 3, 3); B * B' / 3 + 0.5I)
    x_true = [1.3, 0.2, -0.1]
    y = K * x_true + noise_sigma * randn(rng, n_obs)
    model(x) = K * x
    return (problem = RetrievalProblem(model, y, Se_inv, x_prior, Sa_inv), K = K, y = y,
            Se_inv = Se_inv, Sa_inv = Sa_inv, x_prior = collect(x_prior), x_true = x_true)
end

@testset "Gauss-Newton on a linear-Gaussian problem: closed form in one step" begin
    p = phase5_linear_problem()
    K, y, Se, Sa, xa = p.K, p.y, p.Se_inv, p.Sa_inv, p.x_prior
    x_hat = xa + inv(K' * Se * K + Sa) * K' * Se * (y - K * xa)

    one_step = solve(p.problem, GaussNewton(max_iterations = 1))
    @test one_step.status == MaxIterations
    @test isapprox(collect(one_step.x), x_hat; rtol = 1e-10)

    result = @inferred solve(p.problem, GaussNewton())
    @test result isa RetrievalResult
    @test isconcretetype(typeof(result))
    @test result.status == Converged
    @test result.n_iterations == 2
    @test result.rodgers_d2_per_state < 1e-20          # second step is zero up to rounding
    @test length(result.cost_history) == 3
    @test isapprox(collect(result.x), x_hat; rtol = 1e-10)
    @test result.x isa typeof(p.problem.x_prior)
    @test result.x_prior == p.problem.x_prior
    @test result.obs_precision === p.problem.obs_precision
    @test result.prior_precision === p.problem.prior_precision   # issue 7
    @test isapprox(result.posterior_covariance, inv(K' * Se * K + Sa); rtol = 1e-10)
    @test isapprox(result.averaging_kernel, I - result.posterior_covariance * Sa;
                   rtol = 1e-10, atol = 1e-12)
    @test isapprox(result.dfs_total, tr(result.averaging_kernel); rtol = 1e-10)
    @test isapprox(result.prior_influence, 1 - result.dfs_total / 3; rtol = 1e-10)
    r = y - K * collect(result.x)
    d = collect(result.x) - xa
    @test isapprox(result.chi2_obs_per_dof, r' * Se * r / (30 - 3); rtol = 1e-10)
    @test isapprox(result.chi2_total_per_dof, (r' * Se * r + d' * Sa * d) / (30 - 3); rtol = 1e-10)
    @test isapprox(result.y_model, K * collect(result.x); rtol = 1e-12)
    @test result.K ≈ K rtol = 1e-14
    @test isfinite(result.condition_number) && result.condition_number >= 1
end

@testset "Levenberg-Marquardt: fixed and adaptive damping reach the same solution" begin
    p = phase5_linear_problem(seed = 9)
    K, y, Se, Sa, xa = p.K, p.y, p.Se_inv, p.Sa_inv, p.x_prior
    x_hat = xa + inv(K' * Se * K + Sa) * K' * Se * (y - K * xa)
    fixed = solve(p.problem, LevenbergMarquardt(damping = FixedDamping(1.0),
                                                max_iterations = 200, rodgers_tolerance = 1e-20))
    adaptive = solve(p.problem, LevenbergMarquardt(damping = AdaptiveDamping(initial = 1.0),
                                                   max_iterations = 200, rodgers_tolerance = 1e-20))
    marquardt = solve(p.problem, LevenbergMarquardt(scaling = :marquardt, max_iterations = 200,
                                                    rodgers_tolerance = 1e-20))
    for result in (fixed, adaptive, marquardt)
        @test result.status == Converged
        @test isapprox(collect(result.x), x_hat; rtol = 1e-8)
        # cost never increases beyond rounding
        @test all(diff(result.cost_history) .<= 1e-12 * result.cost_history[1])
    end
    @test isapprox(collect(fixed.x), collect(adaptive.x); rtol = 1e-8)
    # fixed damping is slower than Gauss-Newton: (1 + gamma) Sa_inv in the step matrix
    @test fixed.n_iterations > 2
    @test @inferred(solve(p.problem, LevenbergMarquardt())) isa RetrievalResult

    # FixedDamping(gamma) step equals the K' Se K + (1 + gamma) Sa step (profile_inversion)
    gamma = 3.0
    first_step = solve(p.problem, LevenbergMarquardt(damping = FixedDamping(gamma), max_iterations = 1))
    expected = xa + (K' * Se * K + (1 + gamma) * Sa) \ (K' * Se * (y - K * xa))
    @test isapprox(collect(first_step.x), expected; rtol = 1e-10)
    @test first_step.status == MaxIterations

    # gain ratio is exactly 1 on a linear problem (predicted cost at the pre-step state)
    x = xa .+ [0.2, -0.1, 0.05]
    F = K * x
    g = K' * Se * (y - F) - Sa * (x - xa)
    for gamma in (0.0, 0.5, 10.0)
        dx = (K' * Se * K + (1 + gamma) * Sa) \ g
        before = total_cost(y, F, Se, x, xa, Sa)
        after = total_cost(y, K * (x + dx), Se, x + dx, xa, Sa)
        predicted = predicted_cost(y, F, K, x, xa, Se, Sa, dx)
        @test isapprox(predicted, after; rtol = 1e-10)
        @test isapprox(gain_ratio(before, after, predicted), 1.0; atol = 1e-10)
    end
end

@testset "adaptive damping rejects an overshooting step (issue 6)" begin
    # Rosenbrock residuals: the undamped Gauss-Newton step from (-1.2, 1) overshoots
    rosenbrock(x) = [10 * (x.b - x.a^2), 1 - x.a]
    x_prior = ComponentVector(a = -1.2, b = 1.0)
    Se_inv = Diagonal([1.0, 1.0])
    Sa_inv = Diagonal([1e-8, 1e-8])
    problem = RetrievalProblem(rosenbrock, [0.0, 0.0], Se_inv, x_prior, Sa_inv)

    gauss_newton = solve(problem, GaussNewton(max_iterations = 1))
    @test gauss_newton.cost_history[2] > 10 * gauss_newton.cost_history[1]   # overshoot

    lm = solve(problem, LevenbergMarquardt(damping = AdaptiveDamping(initial = 1e-3),
                                           scaling = :marquardt, max_iterations = 100,
                                           rodgers_tolerance = 1e-12))
    @test lm.status == Converged
    m = match(r"adaptive damping rejected (\d+) steps", lm.message)
    @test m !== nothing
    @test parse(Int, m.captures[1]) >= 1
    # only cost-reducing steps are accepted (up to rounding near convergence)
    @test all(diff(lm.cost_history) .<= 1e-12 * lm.cost_history[1])
    @test isapprox(collect(lm.x), [1.0, 1.0]; atol = 1e-4)

    # if no step can reduce the cost the solve fails rather than accepting it
    stuck = solve(problem, LevenbergMarquardt(damping = AdaptiveDamping(initial = 1e-6,
                                                                        max_retries = 0),
                                              scaling = :marquardt))
    @test stuck.status == Failed
    @test occursin("no cost-reducing step", stuck.message)
    @test stuck.n_iterations == 0
end

@testset "noise realizations: chi squared and posterior coverage" begin
    rng = MersenneTwister(2026)
    n_obs, n_state, n_real = 20, 3, 500
    K = randn(rng, n_obs, n_state)
    sigma = 0.1
    Se_inv = Diagonal(fill(1 / sigma^2, n_obs))
    model(x) = K * x
    x_true = [1.0, -0.5, 0.25]
    p_2sigma = 0.9544997361036416                         # P(|z| < 2), standard normal

    # weak prior, fixed truth: chi2_obs_per_dof has expectation 1
    weak = Matrix(1e-8I, n_state, n_state)
    x_prior = ComponentVector(a = 0.0, b = [0.0, 0.0])
    chi2 = zeros(n_real)
    covered = 0
    for i in 1:n_real
        y = K * x_true + sigma * randn(rng, n_obs)
        result = solve(RetrievalProblem(model, y, Se_inv, x_prior, weak), GaussNewton())
        chi2[i] = result.chi2_obs_per_dof
        s = sqrt.(diag(result.posterior_covariance))
        covered += count(abs.(collect(result.x) - x_true) .<= 2 .* s)
    end
    @test abs(mean(chi2) - 1) < 3 * std(chi2) / sqrt(n_real)
    rate = covered / (n_real * n_state)
    @test abs(rate - p_2sigma) < 3 * sqrt(p_2sigma * (1 - p_2sigma) / (n_real * n_state))

    # informative prior, truth drawn from the prior: Bayesian 2-sigma coverage
    Sa = Diagonal([0.04, 0.09, 0.01])
    L = sqrt(Sa)
    covered = 0
    for i in 1:n_real
        truth = collect(x_prior) + L * randn(rng, n_state)
        y = K * truth + sigma * randn(rng, n_obs)
        result = solve(RetrievalProblem(model, y, Se_inv, x_prior, inv(Sa)), GaussNewton())
        s = sqrt.(diag(result.posterior_covariance))
        covered += count(abs.(collect(result.x) - truth) .<= 2 .* s)
    end
    rate = covered / (n_real * n_state)
    @test abs(rate - p_2sigma) < 3 * sqrt(p_2sigma * (1 - p_2sigma) / (n_real * n_state))
end

@testset "zero prior precision: least squares and identity averaging kernel" begin
    p = phase5_linear_problem(seed = 4, prior = :zero)
    x_ls = p.K \ p.y
    for solver in (GaussNewton(),
                   LevenbergMarquardt(max_iterations = 200, rodgers_tolerance = 1e-20),
                   LevenbergMarquardt(damping = FixedDamping(1.0), max_iterations = 200,
                                      rodgers_tolerance = 1e-20))
        result = solve(p.problem, solver)
        @test result.status == Converged
        @test isapprox(collect(result.x), x_ls; rtol = 1e-8)
        @test isapprox(result.averaging_kernel, Matrix(1.0I, 3, 3); atol = 1e-10)
        @test isapprox(result.dfs_total, 3.0; rtol = 1e-10)
        @test abs(result.prior_influence) < 1e-10
        @test result.prior_precision == zeros(3, 3)
        @test result.chi2_obs_per_dof ≈ result.chi2_total_per_dof rtol = 1e-12
    end
end

@testset "status: MaxIterations and Failed" begin
    nonlinear(x) = [exp(x.a) + x.b, x.a * x.b, x.b^3, x.a]
    x_prior = ComponentVector(a = 0.0, b = 1.0)
    y = nonlinear(ComponentVector(a = 0.7, b = 1.6))
    problem = RetrievalProblem(nonlinear, y, Diagonal(fill(1e6, 4)), x_prior, Diagonal([1e-4, 1e-4]))
    capped = solve(problem, GaussNewton(max_iterations = 1, rodgers_tolerance = 1e-12))
    @test capped.status == MaxIterations
    @test capped.n_iterations == 1
    @test occursin("max_iterations", capped.message)
    full = solve(problem, GaussNewton(rodgers_tolerance = 1e-12))
    @test full.status == Converged
    @test isapprox(collect(full.x), [0.7, 1.6]; rtol = 1e-6)

    # non-finite model output after a step
    K = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    clipped(x) = K * x .+ (x.a > 2.5 ? NaN : 0.0)
    problem = RetrievalProblem(clipped, K * [3.0, 1.0], Diagonal(fill(1e4, 3)),
                               ComponentVector(a = 0.0, b = 0.0), zeros(2, 2))
    failed = solve(problem, GaussNewton())
    @test failed.status == Failed
    @test occursin("not finite", failed.message)
    @test all(isnan, failed.posterior_covariance)

    # singular normal matrix after a step: the Jacobian of a^2 vanishes at a = 0, no prior.
    # From a = 1 the step is exactly -1 (normal matrix 4, gradient -4, no rounding).
    quadratic(x) = [x.a^2, x.a^2]
    problem = RetrievalProblem(quadratic, [-1.0, -1.0], Diagonal([0.5, 0.5]),
                               ComponentVector(a = 1.0), zeros(1, 1))
    singular = solve(problem, GaussNewton())
    @test singular.status == Failed
    @test occursin("singular", singular.message)
    @test singular.x.a == 0.0
    @test isnan(singular.dfs_total)
    @test all(isascii, report(singular))

    # invalid solver settings
    @test_throws ArgumentError GaussNewton(max_iterations = 0)
    @test_throws ArgumentError LevenbergMarquardt(scaling = :identity)
    @test_throws ArgumentError FixedDamping(-1.0)
    @test_throws ArgumentError AdaptiveDamping(factor = 1.0)
end
