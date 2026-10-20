# Tests for src/core/diagnostics.jl and the report in src/core/retrieval_result.jl.

# random symmetric positive definite matrix with eigenvalues spread over `spread`
function phase5_random_spd(rng, n; scale = 1.0, spread = 10.0)
    Q = Matrix(qr(randn(rng, n, n)).Q)
    lambda = scale .* exp.(range(0, log(spread); length = n))
    M = Q * Diagonal(lambda) * Q'
    return (M + M') / 2
end

@testset "posterior covariance and averaging kernel identities (issue 1)" begin
    rng = MersenneTwister(20261019)
    n_obs, n_state = 25, 4
    K = randn(rng, n_obs, n_state)
    Se_inv = phase5_random_spd(rng, n_obs; scale = 50.0)
    Sa_inv = phase5_random_spd(rng, n_state; scale = 2.0)

    S = posterior_covariance(K, Se_inv, Sa_inv)
    S_direct = inv(K' * Se_inv * K + Sa_inv)
    @test S isa Matrix{Float64}
    @test isapprox(S, S_direct; rtol = 1e-10)
    @test S ≈ S' rtol = 1e-12

    A = averaging_kernel(K, Se_inv, Sa_inv)
    @test isapprox(A, I - S_direct * Sa_inv; rtol = 1e-10, atol = 1e-12)
    G = gain_matrix(K, Se_inv, Sa_inv)
    @test size(G) == (n_state, n_obs)
    @test isapprox(G * K, A; rtol = 1e-10)
    @test isapprox(dfs_per_element(A), diag(A); rtol = 1e-14)
    @test isapprox(dfs_total(A), tr(I - S_direct * Sa_inv); rtol = 1e-10)
    @test 0 < dfs_total(A) < n_state
    @test isapprox(prior_influence(dfs_total(A), n_state), 1 - tr(A) / n_state; rtol = 1e-12)

    N = normal_matrix(K, Se_inv, Sa_inv)
    @test N isa Symmetric{Float64, Matrix{Float64}}
    @test isapprox(Matrix(N), K' * Se_inv * K + Sa_inv; rtol = 1e-12)
    dx = randn(rng, n_state)
    @test isapprox(rodgers_d2(dx, K, Se_inv, Sa_inv), dx' * (K' * Se_inv * K + Sa_inv) * dx;
                   rtol = 1e-12)
end

@testset "zero prior: least-squares covariance and identity averaging kernel" begin
    rng = MersenneTwister(7)
    K = randn(rng, 12, 3)
    Se_inv = Diagonal(fill(400.0, 12))
    Sa_inv = zeros(3, 3)
    @test isapprox(posterior_covariance(K, Se_inv, Sa_inv), inv(K' * Se_inv * K); rtol = 1e-10)
    @test isapprox(averaging_kernel(K, Se_inv, Sa_inv), Matrix(1.0I, 3, 3); atol = 1e-10)
    @test isapprox(dfs_total(averaging_kernel(K, Se_inv, Sa_inv)), 3.0; rtol = 1e-10)
end

@testset "costs, chi squared and gain ratio helpers" begin
    @test observation_cost([1.0, 2.0], [1.1, 1.9], Diagonal([100.0, 100.0])) ≈ 2.0 rtol = 1e-12
    @test prior_cost([1.0, 3.0], [0.0, 1.0], Diagonal([1.0, 0.25])) ≈ 2.0 rtol = 1e-12
    @test total_cost([1.0], [0.0], Diagonal([1.0]), [2.0], [0.0], Diagonal([1.0])) ≈ 5.0
    @test chi2_per_dof(17.0, 20, 3) ≈ 1.0 rtol = 1e-14
    @test isnan(chi2_per_dof(5.0, 3, 3))
    @test gain_ratio(10.0, 4.0, 4.0) == 1.0
    @test gain_ratio(10.0, 7.0, 4.0) == 0.5
    # predicted cost at dx = 0 is the current cost
    K = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    y, F = [1.0, 2.0, 3.5], [0.5, 1.0, 1.5]
    x, xa = [0.5, 1.0], [0.0, 0.0]
    Se, Sa = Diagonal(fill(4.0, 3)), Diagonal([1.0, 1.0])
    @test predicted_cost(y, F, K, x, xa, Se, Sa, zeros(2)) ≈ total_cost(y, F, Se, x, xa, Sa) rtol = 1e-14
end

@testset "scaled condition number is independent of state units" begin
    rng = MersenneTwister(3)
    K = randn(rng, 30, 3)
    Se_inv = Diagonal(fill(100.0, 30))
    Sa_inv = Diagonal([1.0, 1.0, 1.0])
    c = normal_matrix_condition(K, Se_inv, Sa_inv)
    @test 1 <= c < 1e3
    # rescale state element 1 by 1e19 (molecules/cm2) and element 3 by 1e-6
    D = Diagonal([1e19, 1.0, 1e-6])
    c_scaled = normal_matrix_condition(K / D, Se_inv, D \ Sa_inv / D)
    @test isapprox(c_scaled, c; rtol = 1e-6)
    @test normal_matrix_condition(Symmetric([1.0 1.0; 1.0 1.0])) > 1e15
    @test normal_matrix_condition([1.0 0.0; 0.0 0.0]) == Inf
end

@testset "report: ASCII text with one CSV row per state element" begin
    rng = MersenneTwister(11)
    K = randn(rng, 15, 4)
    x_prior = ComponentVector(CH4 = 1.0, baseline = (window_1 = [0.5, 0.0, 0.0],))
    model(x) = K * x
    y = K * [1.1, 0.4, 0.1, -0.1] + 0.01 * randn(rng, 15)
    problem = RetrievalProblem(model, y, Diagonal(fill(1e4, 15)), x_prior,
                               prior_precision_from_sigma(ComponentVector(CH4 = 1.0,
                                   baseline = (window_1 = [1.0, 1.0, 1.0],))))
    result = solve(problem, GaussNewton())
    @test result.dfs_per_element isa typeof(result.x)
    @test result.dfs_per_element.CH4 ≈ result.averaging_kernel[1, 1] rtol = 1e-14
    @test result.dfs_per_element.baseline.window_1 ≈ diag(result.averaging_kernel)[2:4] rtol = 1e-14
    @test result.dfs_total ≈ tr(result.averaging_kernel) rtol = 1e-12

    text = report(result)
    @test text isa String
    @test all(isascii, text)
    @test !any(c -> 'Ͱ' <= c <= 'Ͽ', text)       # no Greek block characters
    for key in ("status: Converged", "iterations: 2", "DFS total:", "chi2 obs per dof:",
                "chi2 total per dof:", "Rodgers d2 per state:", "prior influence:")
        @test occursin(key, text)
    end
    lines = split(text, '\n')
    first_line = findfirst(==("```csv"), lines)
    @test first_line !== nothing
    last_line = findnext(==("```"), lines, first_line + 1)
    @test last_line !== nothing
    csv = lines[first_line + 1:last_line - 1]
    @test csv[1] == "name,prior,retrieved,posterior_sigma,dfs,ak_diagonal"
    rows = csv[2:end]
    @test length(rows) == length(x_prior) == 4
    @test [split(r, ',')[1] for r in rows] == state_labels(x_prior)
    @test state_labels(x_prior) == ["CH4", "baseline.window_1[1]", "baseline.window_1[2]",
                                    "baseline.window_1[3]"]
    @test all(r -> length(split(r, ',')) == 6, rows)
    sigma_ch4 = parse(Float64, split(rows[1], ',')[4])
    @test isapprox(sigma_ch4, sqrt(result.posterior_covariance[1, 1]); rtol = 1e-5)

    custom = report(result; state_names = ["xch4", "c0", "c1", "c2"])
    @test occursin("\nc1,", custom)
    @test_throws ArgumentError report(result; state_names = ["only_one"])
end
