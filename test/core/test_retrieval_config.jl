# Phase 4: RetrievalConfig and build_problem.

let
    nu = collect(6076.0:0.0067:6084.0)
    cond = (pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
    m = Measurement(y = fill(2.0, length(nu)), coordinate = nu, noise_variance = 1e-6,
                    conditions = cond)
    absorbers = (H2O = Absorber(:H2O, fixture_voigt(:H2O, (6076.0, 6084.0))),
                 CH4 = Absorber(:CH4, fixture_voigt(:CH4, (6076.0, 6084.0))))
    n_air = homogeneous_air_column(850.0, 290.0, 195017.0)
    x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1.9e-6 * n_air,
                              baseline = (window_1 = zeros(3), window_2 = zeros(2)))
    sigma = ComponentVector(H2O = n_air, CH4 = 1e-6 * n_air,
                            baseline = (window_1 = fill(10.0, 3), window_2 = fill(10.0, 2)))

    @testset "configuration validation names the field" begin
        @test_throws r"windows_cm" RetrievalConfig(windows_cm = Tuple{Float64,Float64}[], baseline_coefficients = Int[])
        @test_throws r"baseline_coefficients" RetrievalConfig(windows_cm = [(1.0, 2.0)], baseline_coefficients = [1, 2])
        @test_throws r"mask_mode" RetrievalConfig(windows_cm = [(1.0, 2.0)], baseline_coefficients = [1], mask_mode = :x)
        @test_throws r"baseline_mode" RetrievalConfig(windows_cm = [(1.0, 2.0)], baseline_coefficients = [1], baseline_mode = :x)
    end

    config = RetrievalConfig(windows_cm = [(6077.0, 6079.0), (6080.0, 6082.0)],
                             baseline_coefficients = [3, 2], fine_grid_spacing_cm = 0.005,
                             masked_windows_cm = [(6078.0, 6078.5)])

    @testset "two windows share one state" begin
        problem = build_problem(config, absorbers, m, x_prior, sigma)
        model = problem.forward_model
        @test model isa MultiWindowModel
        @test length(problem.y) == n_observations(model)
        # baseline prior from the measurement: max intensity first, zeros after
        @test problem.x_prior.baseline.window_1 == [2.0, 0.0, 0.0]
        @test problem.x_prior.baseline.window_2 == [2.0, 0.0]
        # masked observations are de-weighted by mask_factor
        w1 = select_window(m, 6077.0, 6079.0)
        masked = findall(c -> 6078.0 < c < 6078.5, w1.coordinate)
        @test all(isapprox.(diag(problem.obs_precision)[masked], 1e-5 / 1e-6; rtol = 1e-12))
        @test diag(problem.obs_precision)[1] == 1 / 1e-6
    end

    @testset "drop mode removes masked observations" begin
        drop = RetrievalConfig(windows_cm = [(6077.0, 6079.0)], baseline_coefficients = [3],
                               masked_windows_cm = [(6078.0, 6078.5)], mask_mode = :drop)
        x1 = ComponentVector(H2O = x_prior.H2O, CH4 = x_prior.CH4, baseline = (window_1 = zeros(3),))
        s1 = ComponentVector(H2O = sigma.H2O, CH4 = sigma.CH4, baseline = (window_1 = fill(10.0, 3),))
        problem = build_problem(drop, absorbers, m, x1, s1)
        n_window = length(select_window(m, 6077.0, 6079.0).y)
        n_masked = count(c -> 6078.0 < c < 6078.5, nu)
        @test length(problem.y) == n_window - n_masked
    end

    @testset "H2O broadening wraps the absorber" begin
        cfg = RetrievalConfig(windows_cm = [(6077.0, 6079.0)], baseline_coefficients = [3],
                              h2o_broadening = Dict(:CH4 => 0.34))
        x1 = ComponentVector(H2O = x_prior.H2O, CH4 = x_prior.CH4, baseline = (window_1 = zeros(3),))
        s1 = ComponentVector(H2O = sigma.H2O, CH4 = sigma.CH4, baseline = (window_1 = fill(10.0, 3),))
        model = build_problem(cfg, absorbers, m, x1, s1).forward_model
        # H2O is in the state, so the broadened CH4 cannot be precomputed
        @test model.absorbers.CH4.model isa H2OBroadened
        @test model.absorbers.H2O.model isa Precomputed
    end

    @testset "retrieve flags must agree with the prior" begin
        cfg = RetrievalConfig(windows_cm = [(6077.0, 6079.0)], baseline_coefficients = [3],
                              retrieve_pressure = true)
        x1 = ComponentVector(H2O = x_prior.H2O, CH4 = x_prior.CH4, baseline = (window_1 = zeros(3),))
        @test_throws r"retrieve_pressure" build_problem(cfg, absorbers, m, x1, x1 .+ 1.0)
    end
end
