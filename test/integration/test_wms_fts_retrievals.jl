# Instruments through the builder and solver: an open-path WMS retrieval with an inverse-crime
# guard (truth from the time-domain DigitalLockIn, retrieval with the quasi-static
# HarmonicExpansion), and an FTS forward model built with its default fine grid.

let
    nu0, gamma = 6046.95, 0.05
    line = AnalyticLorentzLine(nu0, 1.6e-20, gamma / 1000.0)
    atm = HomogeneousPath(pressure_hPa = 1000.0, temperature_K = 295.0, path_length_cm = 1.0e4)
    n_air = dry_air_column(atm)
    absorbers = (CH4 = Absorber(:CH4, line),)

    @testset "WMS retrieval, DigitalLockIn truth and HarmonicExpansion retrieval" begin
        times = collect(range(0.01, 0.19; length = 90))
        m0 = Measurement(y = zeros(length(times)), coordinate = times, noise_variance = 1.0)
        tuning = LaserTuning(LinearRamp(start_cm = nu0 - 0.3, rate_cm_per_s = 3.0),
                             SinusoidalModulation(amplitude_cm = 2.2 * gamma, frequency_Hz = 5e3))
        laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1, linear_phase_rad = 0.0)
        observable = HarmonicProjection(2, 0.0, 0)
        grid = range(nu0 - 0.5, nu0 + 0.5; step = 0.002)
        x_true = ComponentVector(CH4 = 2.0e-6 * n_air, instrument = (linear_amplitude = 0.1,))

        truth_instrument = WavelengthModulation(tuning, laser,
            DigitalLockIn(lowpass = Butterworth(order = 4, cutoff_Hz = 400.0),
                          sampling_rate_Hz = 100e3, pre_roll_s = 0.02), observable)
        truth_model = build_forward_model(atm, absorbers, truth_instrument, NoBaseline(), LinearSpace(),
                                          ColumnDensity(), m0, x_true; wavenumber_fine_cm = grid)
        y = truth_model(x_true)
        sigma_obs = 1e-3 * maximum(abs, y)
        m = Measurement(y = y, coordinate = times, noise_variance = sigma_obs^2)

        retrieval_instrument = WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (2,)), observable)
        x_prior = ComponentVector(CH4 = 1.6e-6 * n_air, instrument = (linear_amplitude = 0.12,))
        prior_sigma = ComponentVector(CH4 = 2.0e-6 * n_air, instrument = (linear_amplitude = 0.05,))
        model = build_forward_model(atm, absorbers, retrieval_instrument, NoBaseline(), LinearSpace(),
                                    ColumnDensity(), m, x_prior; wavenumber_fine_cm = grid)
        @test model.absorbers.CH4.model isa Precomputed
        problem = RetrievalProblem(model, m.y, obs_precision(LinearSpace(), m), x_prior,
                                   prior_precision_from_sigma(prior_sigma, x_prior))
        result = solve(problem, LevenbergMarquardt())
        @test result.status == Converged
        # filter lag in the truth limits agreement; 1 percent is far below the 20 percent prior error
        @test isapprox(result.x.CH4, x_true.CH4; rtol = 0.01)
        @test abs(result.x.CH4 - x_true.CH4) < abs(x_prior.CH4 - x_true.CH4) / 10
        @test result.dfs_per_element.CH4 > 0.99
    end

    @testset "FTS forward model through the builder" begin
        nu_obs = collect(range(nu0 - 0.4, nu0 + 0.4; step = 0.02))
        m = Measurement(y = ones(length(nu_obs)), coordinate = nu_obs, noise_variance = 1e-6)
        x = ComponentVector(CH4 = 2.0e-6 * n_air, baseline = (window_1 = [1.0, 0.0],))
        fts = FTS(max_path_difference_cm = 5.0)
        model = build_forward_model(atm, absorbers, fts, LegendreBaseline(2), LinearSpace(),
                                    ColumnDensity(), m, x)
        y = model(x)
        @test length(y) == length(nu_obs)
        @test minimum(y) < 0.999
        # the instrument smooths the line: depth at the centre is below the monochromatic depth
        tau_centre = x.CH4 * line.strength / (pi * gamma)
        @test 1 - minimum(y) < 1 - exp(-tau_centre)
        K = ForwardDiff.jacobian(model, x)
        K_fd = finite_difference_jacobian(v -> model(ComponentVector(v, getaxes(x))), collect(x))
        @test isapprox(K, K_fd; rtol = 1e-5)
    end
end
