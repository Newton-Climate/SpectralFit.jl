# CovariancePrecision: a banded observation covariance applied by sparse Cholesky solves gives
# the same retrieval and diagnostics as the dense inverse passed as obs_precision.
using JLD2

let
    rng = MersenneTwister(21)
    n_obs = 120
    # banded covariance: lock-in output covariance on closely spaced times, plus a white floor
    lockin = DigitalLockIn(lowpass = Butterworth(order = 4, cutoff_Hz = 400.0), sampling_rate_Hz = 50e3)
    times = collect(range(0.0, 0.06; length = n_obs))
    Se = 3.0 .* lockin_output_covariance(lockin, times) / lockin_output_covariance(lockin, times)[1, 1] * 1e-4
    Se = Se + 1e-7 * I
    Se_dense = Matrix(Se)
    P = CovariancePrecision(Se)

    @testset "construction, products and checks" begin
        @test size(P) == (n_obs, n_obs)
        @test observation_covariance(P) ≈ Se
        v = randn(rng, n_obs)
        M = randn(rng, n_obs, 3)
        @test P * v ≈ Se_dense \ v rtol = 1e-10
        @test P * M ≈ Se_dense \ M rtol = 1e-10
        @test Matrix(P) ≈ inv(Se_dense) rtol = 1e-8
        @test_throws ArgumentError CovariancePrecision(sparse([1.0 0.5; 0.4 1.0]))   # asymmetric
        @test_throws ArgumentError CovariancePrecision(sparse([1.0 2.0; 2.0 1.0]))   # indefinite
        @test_throws ArgumentError CovariancePrecision(ones(2, 3))
        @test_throws ArgumentError CovariancePrecision(sparse([NaN 0.0; 0.0 1.0]))
    end

    # mildly nonlinear model on a nested state
    x_prior = ComponentVector(a = 1.0, b = [0.2, -0.1])
    t = range(0, 1; length = n_obs)
    model(x) = @. x.a * exp(-x.b[1] * t) + x.b[2] * t^2
    x_true = [1.3, 0.5, 0.3]
    L = cholesky(Symmetric(Se_dense)).L
    y = model(ComponentVector(x_true, getaxes(x_prior))) + L * randn(rng, n_obs)
    Sa_inv = Diagonal([1.0, 4.0, 4.0])

    @testset "banded covariance retrieval equals the dense-inverse retrieval" begin
        p_banded = RetrievalProblem(model, y, P, x_prior, Sa_inv)
        p_dense = RetrievalProblem(model, y, Symmetric(inv(Se_dense)) |> Matrix, x_prior, Sa_inv)
        for solver in (GaussNewton(), LevenbergMarquardt())
            r1 = solve(p_banded, solver)
            r2 = solve(p_dense, solver)
            @test r1.status == Converged && r2.status == Converged
            @test isapprox(collect(r1.x), collect(r2.x); rtol = 1e-8)
            @test isapprox(r1.posterior_covariance, r2.posterior_covariance; rtol = 1e-8)
            @test isapprox(r1.averaging_kernel, r2.averaging_kernel; rtol = 1e-7, atol = 1e-10)
            @test isapprox(r1.chi2_obs_per_dof, r2.chi2_obs_per_dof; rtol = 1e-8)
            @test isapprox(r1.condition_number, r2.condition_number; rtol = 1e-6)
            @test r1.obs_precision === P
        end
        r = solve(p_banded, GaussNewton())
        K = r.K
        @test gain_matrix(K, P, Sa_inv) ≈ gain_matrix(K, inv(Se_dense), Sa_inv) rtol = 1e-8
        @test rodgers_d2([0.1, 0.0, 0.2], K, P, Sa_inv) ≈ rodgers_d2([0.1, 0.0, 0.2], K, inv(Se_dense), Sa_inv) rtol = 1e-8
        @test occursin("posterior_sigma", report(r))
        # posterior consistent with the closed form inv(K' Se^-1 K + Sa^-1)
        @test r.posterior_covariance ≈ inv(K' * (Se_dense \ K) + Sa_inv) rtol = 1e-8
        @test_throws ArgumentError RetrievalProblem(model, y, CovariancePrecision(Se[1:end-1, 1:end-1]),
                                                    x_prior, Sa_inv)
    end

    @testset "results file round trip keeps the covariance" begin
        r = solve(RetrievalProblem(model, y, P, x_prior, Sa_inv), GaussNewton())
        path = joinpath(mktempdir(), "banded.jld2")
        save_results(path, [r])
        loaded = only(load_results(path).results)
        @test loaded.obs_precision isa CovariancePrecision
        @test observation_covariance(loaded.obs_precision) ≈ observation_covariance(P)
        @test loaded.posterior_covariance ≈ r.posterior_covariance
    end
end
