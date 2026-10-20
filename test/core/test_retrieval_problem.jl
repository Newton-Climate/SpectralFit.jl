# Tests for src/core/retrieval_problem.jl: validation at construction and the Jacobian.

# Runs `f`, expects an ArgumentError, returns its message ("" if nothing was thrown).
function phase5_error_message(f)
    try
        f()
    catch err
        err isa ArgumentError || rethrow()
        return err.msg
    end
    return ""
end

@testset "RetrievalProblem construction" begin
    x_prior = ComponentVector(a = 1, b = [2, 3])          # integers are converted to Float64
    model(x) = [x.a + x.b[1], x.a * x.b[2], exp(0.1 * x.b[1]), x.a - x.b[2], x.b[1]^2]
    y = model(ComponentVector(a = 1.1, b = [2.0, 2.9]))
    Se_inv = Diagonal(fill(1e4, 5))
    Sa_inv = Diagonal([1.0, 1.0, 1.0])
    problem = RetrievalProblem(model, y, Se_inv, x_prior, Sa_inv)
    @test problem.x_prior isa ComponentVector{Float64}
    @test problem.x_prior.b == [2.0, 3.0]
    @test problem.y == y
    @test problem.obs_precision === Se_inv
    @test problem.prior_precision === Sa_inv
    @test n_state(problem) == 3
    @test n_obs(problem) == 5
    # dense and sparse precisions are accepted
    @test RetrievalProblem(model, y, Matrix(Se_inv), x_prior, Matrix(Sa_inv)) isa RetrievalProblem
    @test RetrievalProblem(model, y, sparse(Se_inv), x_prior, Sa_inv) isa RetrievalProblem
    # a zero prior precision is allowed (least-squares mode, open question 7)
    @test RetrievalProblem(model, y, Se_inv, x_prior, zeros(3, 3)) isa RetrievalProblem
end

@testset "Jacobian from ForwardDiff on ComponentVector states" begin
    x_prior = ComponentVector(gas = 2.0, baseline = (w1 = [1.0, 0.1, -0.05],))
    grid = range(-1, 1; length = 40)
    model(x) = exp.(-x.gas .* exp.(-grid .^ 2 ./ 0.1)) .*
               (x.baseline.w1[1] .+ x.baseline.w1[2] .* grid .+ x.baseline.w1[3] .* grid .^ 2)
    problem = RetrievalProblem(model, model(x_prior), Diagonal(fill(1e6, 40)), x_prior,
                               Diagonal(fill(1e-2, 4)))
    y_model, K = model_and_jacobian(problem, x_prior)
    @test y_model ≈ model(x_prior) rtol = 1e-14
    K_fd = finite_difference_jacobian(v -> model(ComponentVector(v, getaxes(x_prior))),
                                      collect(x_prior))
    @test isapprox(K, K_fd; rtol = 1e-7, atol = 1e-9)
    # the buffer is reused but results are copies
    x2 = copy(x_prior); x2.gas = 1.0
    y2, K2 = model_and_jacobian(problem, x2)
    @test y_model ≈ model(x_prior) rtol = 1e-14
    @test !(K2 ≈ K)
    @test evaluate_model(problem, x2) ≈ model(x2) rtol = 1e-14
end

@testset "RetrievalProblem rejects bad input, naming the argument" begin
    rng = MersenneTwister(5)
    K = randn(rng, 6, 2)
    model(x) = K * x
    x_prior = ComponentVector(a = 0.0, b = 0.0)
    y = randn(rng, 6)
    Se_inv = Diagonal(fill(100.0, 6))
    Sa_inv = Diagonal([1.0, 1.0])
    build(; y = y, Se = Se_inv, x = x_prior, Sa = Sa_inv, f = model) =
        RetrievalProblem(f, y, Se, x, Sa)

    msg = phase5_error_message(() -> build(y = randn(rng, 5)))
    @test startswith(msg, "y:") && occursin("length 5", msg)
    msg = phase5_error_message(() -> build(Se = Diagonal(fill(1.0, 5))))
    @test startswith(msg, "obs_precision:") && occursin("n_obs", msg)
    msg = phase5_error_message(() -> build(Sa = Diagonal(fill(1.0, 3))))
    @test startswith(msg, "prior_precision:") && occursin("n_state", msg)

    msg = phase5_error_message(() -> build(y = [y[1:5]; NaN]))
    @test startswith(msg, "y:") && occursin("non-finite", msg)
    msg = phase5_error_message(() -> build(x = ComponentVector(a = Inf, b = 0.0)))
    @test startswith(msg, "x_prior:")
    msg = phase5_error_message(() -> build(Se = Diagonal([fill(100.0, 5); Inf])))
    @test startswith(msg, "obs_precision:") && occursin("non-finite", msg)

    # not positive definite observation precision: zero, negative, indefinite dense
    msg = phase5_error_message(() -> build(Se = Diagonal([fill(100.0, 5); 0.0])))
    @test startswith(msg, "obs_precision:") && occursin("positive definite", msg)
    indefinite = Matrix(Se_inv); indefinite[1, 2] = indefinite[2, 1] = 500.0
    msg = phase5_error_message(() -> build(Se = indefinite))
    @test startswith(msg, "obs_precision:") && occursin("positive definite", msg)
    nonsymmetric = Matrix(Se_inv); nonsymmetric[1, 2] = 1.0
    msg = phase5_error_message(() -> build(Se = nonsymmetric))
    @test startswith(msg, "obs_precision:") && occursin("symmetric", msg)

    # prior precision must be positive semi-definite (zero allowed, negative not)
    msg = phase5_error_message(() -> build(Sa = Diagonal([1.0, -1.0])))
    @test startswith(msg, "prior_precision:") && occursin("semi-definite", msg)
    msg = phase5_error_message(() -> build(Sa = [1.0 2.0; 2.0 1.0]))
    @test startswith(msg, "prior_precision:") && occursin("semi-definite", msg)

    # model output not finite at the prior
    msg = phase5_error_message(() -> build(f = x -> K * x .+ (x.a == 0 ? NaN : 0.0)))
    @test startswith(msg, "forward_model:")

    # condition number above 1e12: two nearly collinear Jacobian columns, (almost) no prior
    K_bad = hcat(K[:, 1], K[:, 1] .* (1 + 1e-9))
    f_bad(x) = K_bad * x
    msg = phase5_error_message(() -> build(f = f_bad, Sa = Diagonal([1e-12, 1e-12])))
    @test occursin("condition number", msg) && occursin("1e12", msg)
    @test occursin("prior_precision", msg) && occursin("forward_model", msg)
    # exactly singular with a zero prior
    msg = phase5_error_message(() -> build(f = x -> K[:, 1] .* (x.a + x.b), Sa = zeros(2, 2)))
    @test occursin("condition number", msg)
    # an informative prior removes the degeneracy
    @test build(f = f_bad, Sa = Diagonal([1.0, 1.0])) isa RetrievalProblem
end

@testset "prior whose data is itself a ComponentVector" begin
    x = ComponentVector(a = 1.0, b = [2.0, 3.0])
    nested = ComponentVector(collect(x), getaxes(x))      # collect keeps a ComponentVector
    f(v) = [v[1], v[2] + v[3], v[3]]
    problem = RetrievalProblem(f, [1.0, 5.0, 3.0], Matrix(1.0I, 3, 3), nested, Matrix(1e-6I, 3, 3))
    result = solve(problem, GaussNewton())
    @test getdata(result.x) isa Vector{Float64}
    @test result.dfs_per_element.b isa AbstractVector
end
