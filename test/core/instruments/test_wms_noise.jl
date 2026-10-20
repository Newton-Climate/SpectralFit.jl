# Banded WMS noise covariance: filter autocorrelation, the lock-in output covariance against a
# Monte Carlo of white noise through DigitalLockIn, and first-order propagation to observables.

let
    fs, fc, f_mod = 100e3, 400.0, 5e3
    lowpass = Butterworth(order = 4, cutoff_Hz = fc)
    lockin = DigitalLockIn(harmonics = (1, 2), lowpass = lowpass, sampling_rate_Hz = fs, pre_roll_s = 0.02)
    tuning = LaserTuning(FixedCenter(6046.95), SinusoidalModulation(amplitude_cm = 0.1, frequency_Hz = f_mod))
    laser = IntensityModulation(mean = 1.0)
    times = collect(range(0.02, 0.06; length = 81))     # 0.5 ms apart, inside the correlation time
    K = length(times)

    @testset "lowpass autocorrelation matches the brute-force sum" begin
        sos = design_filter(lowpass, fs)
        R = lowpass_autocorrelation(sos)
        h = filter_signal(sos, [1.0; zeros(20_000)])
        for d in (0, 1, 5, 40, 200)
            @test isapprox(R[d+1], dot(h[1:end-d], h[1+d:end]); rtol = 1e-10, atol = 1e-16)
        end
        # noise-equivalent bandwidth: R(0) = sum h^2 = (2 / fs) * int_0^inf |H|^2 df ~ 2 B / fs
        f = range(0, fs / 2; length = 200_001)
        bandwidth = sum(abs2.(frequency_response.(Ref(sos), f))) * step(f)
        @test isapprox(R[1], 2bandwidth / fs; rtol = 1e-3)
    end

    @testset "lockin_output_covariance: banded, symmetric, tolerance controls the band" begin
        C = lockin_output_covariance(lockin, times)
        @test issymmetric(C)
        @test isposdef(Matrix(C))
        bandwidth(M) = maximum(abs(i - j) for (i, j, _) in zip(findnz(M)...))
        C_loose = lockin_output_covariance(lockin, times; band_tolerance = 1e-2)
        @test bandwidth(C_loose) < bandwidth(C) < K - 1
        # unsorted times give the permuted matrix
        p = [1; 1 .+ randperm(MersenneTwister(3), K - 1)]   # the first time anchors the grid
        @test lockin_output_covariance(lockin, times[p]) ≈ C[p, p]
        @test_throws ArgumentError lockin_output_covariance(lockin, times; band_tolerance = 0.0)
    end

    @testset "Monte Carlo: white noise through DigitalLockIn matches the banded model" begin
        instrument = WavelengthModulation(tuning, laser, lockin, Quadratures((1, 2)))
        m = Measurement(y = zeros(4K), coordinate = repeat(times, 4), noise_variance = 1.0)
        p = SF.CoreLayer.prepare(instrument, m)
        sigma2 = 0.7
        Se = wms_noise_covariance(p, sigma2)
        @test size(Se) == (4K, 4K) && issymmetric(Se)
        n_samples = length(lockin_sample_times(p))
        rng = MersenneTwister(11)
        n_mc = 3000
        q = zeros(4K, n_mc)
        dc = zeros(K, n_mc)
        for s in 1:n_mc
            h = demodulate(p, sqrt(sigma2) .* randn(rng, n_samples))
            q[:, s] = evaluate_observable(Quadratures((1, 2)), h, nothing)
            dc[:, s] = dc_level(h)
        end
        E = cov(q; dims = 2)
        mid = 20:60
        v0 = mean(Se[i, i] for i in mid)
        # averaged over times, the sampling error of a covariance is ~ v0 * sqrt(2 / n_mc) / sqrt(n_indep)
        tol = 0.03 * v0
        for block in 0:3, lag in 0:4
            model = mean(Se[block*K+i, block*K+i+lag] for i in mid)
            empirical = mean(E[block*K+i, block*K+i+lag] for i in mid)
            @test abs(model - empirical) < tol
        end
        # mixing gain 2 for the quadratures, 1 for DC
        C = lockin_output_covariance(lockin, times)
        @test isapprox(v0, 2sigma2 * C[40, 40]; rtol = 1e-12)
        @test abs(mean(var(dc; dims = 2)[mid]) - sigma2 * C[40, 40]) < 0.03 * sigma2 * C[40, 40]
        # X and Y of a harmonic, and different harmonics, are uncorrelated (model and data)
        @test all(iszero, Se[1:K, K+1:2K]) && all(iszero, Se[1:K, 2K+1:4K])
        for lag in 0:3
            @test abs(mean(E[i, K+i+lag] for i in mid)) < tol
            @test abs(mean(E[i, 2K+i+lag] for i in mid)) < tol
        end
    end

    @testset "ratio observable: first-order propagation against Monte Carlo" begin
        observable = HarmonicProjection(2, 0.0, 0)          # X_2 / DC
        instrument = WavelengthModulation(tuning, laser, lockin, observable)
        m = Measurement(y = zeros(K), coordinate = times, noise_variance = 1.0)
        p = SF.CoreLayer.prepare(instrument, m)
        t = lockin_sample_times(p)
        w = 2pi * f_mod
        clean = @. 1.0 + 0.1 * cos(w * t) + 0.05 * cos(2w * t) + 0.02 * sin(2w * t)
        signals = demodulate(p, clean)
        sigma2 = 0.05^2
        Se = wms_noise_covariance(p, sigma2; signals = signals)
        rng = MersenneTwister(5)
        n_mc = 3000
        o = reduce(hcat, [evaluate_observable(observable, demodulate(p, clean .+ sqrt(sigma2) .* randn(rng, length(t))), nothing)
                          for _ in 1:n_mc])
        E = cov(o; dims = 2)
        mid = 20:60
        v0 = mean(Se[i, i] for i in mid)
        for lag in 0:3
            @test abs(mean(Se[i, i+lag] for i in mid) - mean(E[i, i+lag] for i in mid)) < 0.04 * v0
        end
        # the DC term of the linearisation matters: ignoring it changes the variance
        J = observable_signal_jacobian(observable, signals, nothing)
        @test size(J) == (K, 5)
        @test all(J[:, 2] .≈ 1 ./ signals.dc)                # d(X2/DC)/dX2
        @test all(J[:, [1, 3, 4]] .== 0)
        # ratio observables need the linearisation point
        @test_throws ArgumentError wms_noise_covariance(p, sigma2)
        @test_throws ArgumentError wms_noise_covariance(p, -1.0; signals = signals)
    end

    @testset "argument checks" begin
        expansion = WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (2,)), Quadratures((2,)))
        m = Measurement(y = zeros(2K), coordinate = repeat(times, 2), noise_variance = 1.0)
        p = SF.CoreLayer.prepare(expansion, m)
        @test_throws ArgumentError wms_noise_covariance(p, 1.0)     # no lock-in to read
        @test size(wms_noise_covariance(p, lockin, 1.0)) == (2K, 2K)
        slow = DigitalLockIn(lowpass = Butterworth(order = 2, cutoff_Hz = 3e3), sampling_rate_Hz = fs)
        @test_throws ArgumentError wms_noise_covariance(p, slow, 1.0)
        bad = HarmonicSignals((2,), ones(K - 1, 1), ones(K - 1, 1), ones(K - 1))
        @test_throws ArgumentError wms_noise_covariance(p, lockin, 1.0; signals = bad)
    end
end
