# WMS retrieval with noise correlated by the lock-in low-pass. Noise-free quadratures come from
# the retrieval model itself (HarmonicExpansion), so the only error is the noise; the noise is
# white detector noise demodulated through a DigitalLockIn, sampled more densely than the
# filter correlation time. Over several seeds the banded covariance gives honest posterior
# sigmas (normalized errors of unit variance, chi2 near 1); a diagonal covariance with the
# same marginal variances reports sigmas that are too small.

let
    nu0, gamma = 6046.95, 0.05
    line = AnalyticLorentzLine(nu0, 1.6e-20, gamma / 1000.0)
    atm = HomogeneousPath(pressure_hPa = 1000.0, temperature_K = 295.0, path_length_cm = 1.0e4)
    n_air = dry_air_column(atm)
    absorbers = (CH4 = Absorber(:CH4, line),)

    @testset "WMS retrieval: banded covariance gives honest posterior sigma" begin
        times = collect(range(0.02, 0.18; length = 321))      # 0.5 ms apart
        K = length(times)
        observable = Quadratures((2,))
        m0 = Measurement(y = zeros(2K), coordinate = repeat(times, 2), noise_variance = 1.0)
        tuning = LaserTuning(LinearRamp(start_cm = nu0 - 0.3, rate_cm_per_s = 3.0),
                             SinusoidalModulation(amplitude_cm = 2.2 * gamma, frequency_Hz = 5e3))
        laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1)
        grid = range(nu0 - 0.6, nu0 + 0.6; step = 0.002)
        x_true = ComponentVector(CH4 = 2.0e-6 * n_air, instrument = (linear_amplitude = 0.1,))
        instrument = WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (2,)), observable)
        model = build_forward_model(atm, absorbers, instrument, NoBaseline(), LinearSpace(),
                                    ColumnDensity(), m0, x_true; wavenumber_fine_cm = grid)
        y_clean = model(x_true)

        lockin = DigitalLockIn(harmonics = (2,), lowpass = Butterworth(order = 4, cutoff_Hz = 400.0),
                               sampling_rate_Hz = 100e3, pre_roll_s = 0.02)
        noise_lockin = SF.CoreLayer.prepare(WavelengthModulation(tuning, laser, lockin, observable), m0)
        n_samples = length(lockin_sample_times(noise_lockin))
        sigma_detector = 0.02 * maximum(abs, y_clean) / sqrt(2 * lowpass_autocorrelation(
            design_filter(lockin.lowpass, lockin.sampling_rate_Hz))[1])
        Se = wms_noise_covariance(model.instrument, lockin, sigma_detector^2)
        P_banded = CovariancePrecision(Se)
        P_diagonal = Diagonal(1 ./ diag(Se))                 # same marginal variances, no correlation

        x_prior = ComponentVector(CH4 = 1.6e-6 * n_air, instrument = (linear_amplitude = 0.12,))
        prior_sigma = ComponentVector(CH4 = 2.0e-5 * n_air, instrument = (linear_amplitude = 0.5,))
        Sa_inv = prior_precision_from_sigma(prior_sigma, x_prior)

        rng = MersenneTwister(2024)
        z_banded, z_diagonal, chi2_banded, sigma_ratio = Float64[], Float64[], Float64[], Float64[]
        for _ in 1:12
            noise = evaluate_observable(observable,
                demodulate(noise_lockin, sigma_detector .* randn(rng, n_samples)), nothing)
            y = y_clean .+ noise
            r_b = solve(RetrievalProblem(model, y, P_banded, x_prior, Sa_inv), LevenbergMarquardt())
            r_d = solve(RetrievalProblem(model, y, P_diagonal, x_prior, Sa_inv), LevenbergMarquardt())
            @test r_b.status == Converged && r_d.status == Converged
            s_b = sqrt(r_b.posterior_covariance[1, 1])
            s_d = sqrt(r_d.posterior_covariance[1, 1])
            push!(z_banded, (r_b.x.CH4 - x_true.CH4) / s_b)
            push!(z_diagonal, (r_d.x.CH4 - x_true.CH4) / s_d)
            push!(chi2_banded, r_b.chi2_obs_per_dof)
            push!(sigma_ratio, s_b / s_d)
        end
        # the diagonal model ignores the positive correlation and is overconfident
        @test all(>(1.3), sigma_ratio)            # about 1.56 here
        # banded: normalized errors have unit variance (12 samples: generous chi2 bounds) and the
        # observation chi2 per dof is close to 1
        @test 0.3 < mean(z_banded .^ 2) < 2.2
        @test mean(z_diagonal .^ 2) > 2 * mean(z_banded .^ 2)
        @test all(c -> 0.75 < c < 1.25, chi2_banded)
    end
end
