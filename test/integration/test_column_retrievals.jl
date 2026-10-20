# Phase 6: end-to-end column retrievals on the HITRAN subset (CH4 window near 6080 cm-1).
#
# Inverse-crime guard: every synthetic measurement here is made by its own forward model on a
# fine grid ten times finer (0.0005 cm-1) than the retrieval grid (0.005 cm-1), so the
# retrieval model cannot reproduce the measurement exactly by construction.

let
    range_cm = (6076.0, 6084.0)
    absorbers = (H2O = Absorber(:H2O, fixture_voigt(:H2O, range_cm)),
                 CH4 = Absorber(:CH4, fixture_voigt(:CH4, range_cm)))
    p_true, t_true, path_cm = 850.0, 290.0, 195017.0
    n_air = homogeneous_air_column(p_true, t_true, path_cm)
    nu = collect(6078.0:0.0067:6082.0)
    conditions = (pressure_hPa = p_true, temperature_K = t_true, path_length_cm = path_cm)
    retrieval_spacing_cm = 0.005
    truth_grid = 6076.5:(retrieval_spacing_cm / 10):6083.5
    window = (6077.9, 6082.1)
    shape_true = [1.0, 0.02, -0.01, 0.005]

    x_true = ComponentVector(H2O = 0.01 * n_air, CH4 = 1900e-9 * n_air, pressure_hPa = p_true,
                             temperature_K = t_true, baseline = (window_1 = shape_true,))
    # perturbed prior (also the first guess); the baseline entries are replaced from the
    # measurement by build_problem (baseline_prior = :from_measurement)
    x_prior = ComponentVector(H2O = 0.012 * n_air, CH4 = 1750e-9 * n_air, pressure_hPa = 830.0,
                              temperature_K = 283.0, baseline = (window_1 = zeros(4),))

    # Truth spectrum in instrument units from a separate model on the fine truth grid, with
    # pressure and temperature fixed at the true values (cross sections precomputed there).
    function truth_spectrum(space)
        atm = HomogeneousPath(pressure_hPa = p_true, temperature_K = t_true, path_length_cm = path_cm)
        x = ComponentVector(H2O = x_true.H2O, CH4 = x_true.CH4, baseline = (window_1 = shape_true,))
        m = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1.0)
        model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(4), space,
                                    ColumnDensity(), m, x; wavenumber_fine_cm = truth_grid)
        y = model(x)
        return space isa LogSpace ? exp.(y) : y
    end
    y_clean = truth_spectrum(LinearSpace())

    config(; kwargs...) = RetrievalConfig(; windows_cm = [window], baseline_coefficients = [4],
                                          fine_grid_spacing_cm = retrieval_spacing_cm,
                                          retrieve_pressure = true, retrieve_temperature = true,
                                          kwargs...)
    measurement(y, noise_variance) = Measurement(y = y, coordinate = nu,
                                                 noise_variance = noise_variance,
                                                 conditions = conditions)
    relative_error(x) = ComponentVector(abs.(collect(x) .- collect(x_true)) ./ abs.(collect(x_true)),
                                        getaxes(x_true))

    # Weak prior for the noise-free case: with noise variance 1e-10 the posterior sigma of CH4
    # is about 0.7 percent, so the prior pulls CH4 by about (0.007 / 1.05)^2 * 8 percent,
    # 4e-6 relative, well below the 1e-4 test tolerance.
    sigma_weak = ComponentVector(H2O = 0.01 * n_air, CH4 = 2000e-9 * n_air, pressure_hPa = 100.0,
                                 temperature_K = 30.0, baseline = (window_1 = fill(1.0, 4),))
    noise_free = build_problem(config(), absorbers, measurement(y_clean, 1e-10), x_prior, sigma_weak)

    @testset "inverse-crime guard: the retrieval model does not reproduce the truth" begin
        model = noise_free.forward_model
        @test step(model.wavenumber_fine_cm) == 10 * step(truth_grid)
        mismatch = maximum(abs, model(x_true) - y_clean)
        @test mismatch > 1e-9           # not identical by construction
        @test mismatch < 1e-4           # but the retrieval grid is adequate
    end

    @testset "noise-free retrieval of H2O, CH4, pressure, temperature and baseline" begin
        result = solve(noise_free, GaussNewton(rodgers_tolerance = 1e-6))
        @test result.status == Converged
        @test all(relative_error(result.x) .< 1e-4)
        # the prior was really perturbed
        @test relative_error(noise_free.x_prior).CH4 > 0.05
        @test relative_error(noise_free.x_prior).pressure_hPa > 0.02
        @test relative_error(noise_free.x_prior).temperature_K > 0.02
    end

    @testset "noise-free retrieval in log space" begin
        # Truth made in LogSpace (log of the instrument signal plus additive Legendre baseline
        # in log space), stored as a linear intensity, retrieved in LogSpace.
        # The same shape coefficients are used, now as an additive baseline in log space.
        y_log_truth = truth_spectrum(LogSpace())
        problem = build_problem(config(measurement_space = LogSpace()), absorbers,
                                measurement(y_log_truth, 1e-10), x_prior, sigma_weak)
        result = solve(problem, GaussNewton(rodgers_tolerance = 1e-6))
        @test result.status == Converged
        @test all(relative_error(result.x) .< 1e-4)
    end

    @testset "noisy retrieval: chi2 and CH4 within 4 posterior sigma" begin
        noise_sigma = 2e-4
        y_noisy = y_clean .+ noise_sigma .* randn(MersenneTwister(2024), length(y_clean))
        sigma = ComponentVector(H2O = 0.005 * n_air, CH4 = 500e-9 * n_air, pressure_hPa = 50.0,
                                temperature_K = 15.0, baseline = (window_1 = fill(1.0, 4),))
        problem = build_problem(config(), absorbers, measurement(y_noisy, noise_sigma^2),
                                x_prior, sigma)
        result = solve(problem, GaussNewton())
        @test result.status == Converged
        @test 0.7 < result.chi2_obs_per_dof < 1.3
        @test 0.7 < result.chi2_total_per_dof < 1.3
        sigma_ch4 = sqrt(result.posterior_covariance[2, 2])
        @test labels(result.x)[2] == "CH4"
        @test abs(result.x.CH4 - x_true.CH4) < 4 * sigma_ch4
        @test sigma_ch4 < sigma.CH4                 # the measurement adds information
        @test 0 < result.dfs_per_element.CH4 <= 1
    end

    @testset "masked windows lower the precision by mask_factor" begin
        masked = [(6079.0, 6079.4), (6080.5, 6080.6)]
        factor = 1e-3
        for space in (LinearSpace(), LogSpace())
            m = measurement(y_clean, fill(4e-8, length(nu)))
            plain = build_problem(config(measurement_space = space), absorbers, m, x_prior, sigma_weak)
            with_mask = build_problem(config(measurement_space = space, masked_windows_cm = masked,
                                             mask_factor = factor),
                                      absorbers, m, x_prior, sigma_weak)
            inside = [lo < c < hi for c in select_window(m, window...).coordinate,
                      (lo, hi) in masked]
            is_masked = vec(any(inside; dims = 2))
            @test count(is_masked) > 10
            p0 = diag(plain.obs_precision)
            p1 = diag(with_mask.obs_precision)
            @test isapprox(p1[is_masked], factor .* p0[is_masked]; rtol = 1e-12)
            @test p1[.!is_masked] == p0[.!is_masked]
            # log space precision propagates the stored linear intensity: y^2 / variance
            y_window = select_window(m, window...).y
            expected = space isa LogSpace ? y_window .^ 2 ./ 4e-8 : fill(1 / 4e-8, length(y_window))
            @test isapprox(p0, expected; rtol = 1e-12)
            @test with_mask.y == plain.y
        end
    end

    @testset "solve is type stable" begin
        @test @inferred(solve(noise_free, GaussNewton())) isa RetrievalResult
    end
end
