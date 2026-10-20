# WMS demodulators: HarmonicExpansion against closed forms, DigitalLockIn against known
# harmonics and against HarmonicExpansion, and ForwardDiff through both.
#
# Closed-form Lorentzian harmonics (Arndt 1965), derived from the complex form. With
# x = (nu - nu0) / gamma and modulation index m = a / gamma, the normalized Lorentzian is
# L(x) = 1 / (1 + x^2) = Re[1 / (1 - i x)], so along the modulation x = x0 + m cos(theta)
#     L = Re[1 / (A + B cos theta)],  A = 1 - i x0,  B = -i m.
# For complex A, B the expansion 1 / (A + B cos theta) = (1/s) [1 + 2 sum_n r^n cos(n theta)]
# holds with s = sqrt(A^2 - B^2) and r = (s - A) / B, choosing the sign of s so |r| < 1 (the
# two candidate r are reciprocal). Hence the cosine Fourier coefficients of L are
#     H_0 = Re(1 / s),  H_n = Re(2 r^n / s),  n >= 1,
# which is our X_n convention (X_n = (1/pi) int S cos(n theta) dtheta). Check: at line centre
# H_2(0, m) = -(2 / m^2) [(2 + m^2) / sqrt(1 + m^2) - 2], the classic Arndt result, and for
# small m H_2 -> (m^2 / 4) L''(x0). For a thin line T = 1 - alpha L the detector signal with
# unit intensity has X_n = -alpha H_n, Y_n = 0 and DC = 1 - alpha H_0.

function wms_arndt(n, x0, m)
    A = complex(1.0, -x0)
    B = complex(0.0, -m)
    s = sqrt(A^2 - B^2)
    r = (s - A) / B
    if abs(r) > 1
        s = -s
        r = (s - A) / B
    end
    return n == 0 ? real(1 / s) : real(2 * r^n / s)
end

wms_measurement(times) = Measurement(y = zeros(length(times)), coordinate = times, noise_variance = 1.0)
wms_blocks(times, nblocks) = Measurement(y = zeros(nblocks * length(times)),
                                         coordinate = repeat(times, nblocks), noise_variance = 1.0)

const WMS_NU0 = 6046.95
const WMS_GAMMA = 0.05

@testset "Arndt closed form" begin
    for m in (0.1, 0.5, 1.0, 2.2, 3.0)
        # the line-centre formula itself cancels to ~1e-12 at m = 0.1
        @test wms_arndt(2, 0.0, m) ≈ -(2 / m^2) * ((2 + m^2) / sqrt(1 + m^2) - 2) rtol = 1e-10
        @test abs(wms_arndt(1, 0.0, m)) < 1e-15
        # direct quadrature of the Lorentzian over the modulation
        th = range(0, 2pi; length = 4097)[1:end-1]
        for x0 in (-2.0, 0.3, 1.7)
            Lth = @. 1 / (1 + (x0 + m * cos(th))^2)
            for n in 0:3
                q = n == 0 ? sum(Lth) / length(th) : 2 * sum(Lth .* cos.(n .* th)) / length(th)
                @test wms_arndt(n, x0, m) ≈ q atol = 1e-13
            end
        end
    end
end

@testset "HarmonicExpansion: Lorentzian harmonics" begin
    alpha = 1e-3
    grid = range(WMS_NU0 - 0.6, WMS_NU0 + 0.6; step = WMS_GAMMA / 200)
    transmission = @. 1 - alpha / (1 + ((grid - WMS_NU0) / WMS_GAMMA)^2)
    times = collect(range(0.0, 1.0; length = 81))
    scan = LinearRamp(start_cm = WMS_NU0 - 4WMS_GAMMA, rate_cm_per_s = 8WMS_GAMMA)
    x0 = @. (times * 8WMS_GAMMA - 4WMS_GAMMA) / WMS_GAMMA
    for m in (0.1, 0.3, 1.0, 2.2, 3.0)
        instrument = WavelengthModulation(LaserTuning(scan, SinusoidalModulation(m * WMS_GAMMA, 10e3)),
                                          IntensityModulation(mean = 1.0),
                                          HarmonicExpansion(harmonics = (1, 2), n_phase_points = 128),
                                          Quadratures((1, 2)))
        p = prepare(instrument, wms_blocks(times, 4))
        @test n_observations(p) == 4 * 81
        h = harmonic_signals(p, grid, transmission, nothing)
        for n in (1, 2)
            expected = -alpha .* wms_arndt.(n, x0, m)
            @test maximum(abs, in_phase(h, n) .- expected) / maximum(abs, expected) < 1e-8
            @test maximum(abs, quadrature(h, n)) < 1e-14
        end
        @test maximum(abs, dc_level(h) .- (1 .- alpha .* wms_arndt.(0, x0, m))) < 1e-8 * alpha
        y = apply_instrument(p, grid, transmission, nothing)
        @test y[1:81] == in_phase(h, 1)
        @test y[82:162] == quadrature(h, 1)
        @test y[163:243] == in_phase(h, 2)
        @test y[244:324] == quadrature(h, 2)
    end
    lo, hi = wavenumber_coverage_cm(prepare(WavelengthModulation(
        LaserTuning(scan, SinusoidalModulation(0.07, 10e3)), IntensityModulation(),
        HarmonicExpansion(), HarmonicRatio(2, 1)), wms_measurement(times)), nothing)
    @test lo ≈ WMS_NU0 - 4WMS_GAMMA - 0.07 && hi ≈ WMS_NU0 + 4WMS_GAMMA - 0.07 + 0.14
end

@testset "HarmonicExpansion: small-modulation limit" begin
    line = AnalyticLorentzLine(WMS_NU0, 2e-21, WMS_GAMMA / 1013.25)
    cond = GasConditions(1013.25, 296.0, 0.0)
    column = 2e19
    T(nu) = exp(-column * cross_section(line, nu, cond))
    grid = range(WMS_NU0 - 0.5, WMS_NU0 + 0.5; step = WMS_GAMMA / 1000)
    transmission = T.(grid)
    times = collect(range(0.0, 1.0; length = 41))
    a = 0.01 * WMS_GAMMA
    scan = LinearRamp(WMS_NU0 - 3WMS_GAMMA, 6WMS_GAMMA)
    instrument = WavelengthModulation(LaserTuning(scan, SinusoidalModulation(a, 10e3)),
                                      IntensityModulation(), HarmonicExpansion(), Quadratures((2,)))
    h = harmonic_signals(prepare(instrument, wms_blocks(times, 2)), grid, transmission, nothing)
    nu = WMS_NU0 .- 3WMS_GAMMA .+ 6WMS_GAMMA .* times
    second = [ForwardDiff.derivative(v -> ForwardDiff.derivative(T, v), n) for n in nu]
    expected = a^2 / 4 .* second
    @test maximum(abs, in_phase(h, 2) .- expected) / maximum(abs, expected) < 1e-3
end

@testset "DigitalLockIn: known harmonics" begin
    f, fs = 5e3, 100e3
    times = collect(range(0.15, 0.3; length = 16))
    instrument = WavelengthModulation(LaserTuning(FixedCenter(WMS_NU0), SinusoidalModulation(0.05, f)),
                                      IntensityModulation(),
                                      DigitalLockIn(harmonics = (1, 2, 3), sampling_rate_Hz = fs),
                                      Quadratures((1, 2, 3)))
    p = prepare(instrument, wms_blocks(times, 6))
    ts = lockin_sample_times(p)
    @test first(ts) == 0.15 && last(ts) >= 0.3
    # synthetic detector trace starting at the first sample (zero filter state there)
    D, R, phase = 2.0, (0.3, 0.07, 0.01), (0.4, -1.2, 2.5)
    trace = [D + sum(R[n] * cos(n * 2pi * f * t - phase[n]) for n in 1:3) for t in ts]
    h = demodulate(p, trace)
    # the filter starts from zero at t = 0.15 s, so compare only after it settles (~0.1 s)
    late = findall(>=(0.27), times)
    for n in 1:3
        @test maximum(abs, in_phase(h, n)[late] .- R[n] * cos(phase[n])) < 1e-6
        @test maximum(abs, quadrature(h, n)[late] .- R[n] * sin(phase[n])) < 1e-6
    end
    @test maximum(abs, dc_level(h)[late] .- D) < 1e-6
    # with a pre-roll the whole record is settled
    settled = WavelengthModulation(instrument.tuning, IntensityModulation(),
                                   DigitalLockIn(harmonics = (1, 2, 3), sampling_rate_Hz = fs, pre_roll_s = 0.15),
                                   Quadratures((1, 2, 3)))
    ps = prepare(settled, wms_blocks(times, 6))
    trace = [D + sum(R[n] * cos(n * 2pi * f * t - phase[n]) for n in 1:3) for t in lockin_sample_times(ps)]
    hs = demodulate(ps, trace)
    for n in 1:3
        @test maximum(abs, magnitude(hs, n) .- R[n]) < 1e-6 * maximum(R)
    end
    @test_throws DimensionMismatch demodulate(ps, trace[1:end-1])
    @test_throws ArgumentError WavelengthModulation(instrument.tuning, IntensityModulation(),
                                                    DigitalLockIn(sampling_rate_Hz = 15e3), HarmonicRatio(2, 1))
end

@testset "DigitalLockIn agrees with HarmonicExpansion on a slow scan" begin
    # fs = 20 f: the detector signal's harmonics alias only from n >= 18, which at modulation
    # index 1 (|r| = sqrt(2) - 1 per harmonic) is below 1e-7 of the signal
    f, fs = 2e3, 40e3
    lowpass = Butterworth(order = 5, cutoff_Hz = 100.0)
    sos = design_filter(lowpass, fs)
    # group delay at zero frequency from the phase slope of the designed filter
    df = 1e-3
    lag = -angle(frequency_response(sos, df)) / (2pi * df)
    @test lag ≈ (2sin(pi / 10) + 2sin(3pi / 10) + 1) / (2pi * 100.0) rtol = 1e-3   # analogue value

    line = AnalyticLorentzLine(WMS_NU0, 2e-21, WMS_GAMMA / 1013.25)
    cond = GasConditions(1013.25, 296.0, 0.0)
    grid = range(WMS_NU0 - 0.4, WMS_NU0 + 0.4; step = WMS_GAMMA / 100)
    transmission = exp.(-1e19 .* cross_section(line, grid, cond))
    rate = WMS_GAMMA                      # one line width per second: slow against 100 Hz
    tuning = LaserTuning(LinearRamp(WMS_NU0 - 3WMS_GAMMA, rate), SinusoidalModulation(1.0WMS_GAMMA, f))
    laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1, linear_phase_rad = 0.3)
    times = collect(range(0.5, 5.5; length = 51))
    lockin = WavelengthModulation(tuning, laser, DigitalLockIn(lowpass = lowpass, sampling_rate_Hz = fs,
                                                               pre_roll_s = 0.3), Quadratures((1, 2)))
    expansion = WavelengthModulation(tuning, laser, HarmonicExpansion(), Quadratures((1, 2)))
    hl = harmonic_signals(prepare(lockin, wms_blocks(times, 4)), grid, transmission, nothing)
    he = harmonic_signals(prepare(expansion, wms_blocks(times, 4)), grid, transmission, nothing)
    he_lag = harmonic_signals(prepare(expansion, wms_blocks(times .- lag, 4)), grid, transmission, nothing)
    dt = 1e-3
    he_p = harmonic_signals(prepare(expansion, wms_blocks(times .+ dt, 4)), grid, transmission, nothing)
    he_m = harmonic_signals(prepare(expansion, wms_blocks(times .- dt, 4)), grid, transmission, nothing)
    for (get, n) in ((h -> in_phase(h, 1), 1), (h -> quadrature(h, 1), 1),
                     (h -> in_phase(h, 2), 2), (h -> quadrature(h, 2), 2), (dc_level, 0))
        rate_of_change = maximum(abs, (get(he_p) .- get(he_m)) ./ (2dt))
        tolerance = 1.2 * lag * rate_of_change + 1e-6
        @test maximum(abs, get(hl) .- get(he)) < tolerance
        @test maximum(abs, get(hl) .- get(he_lag)) < 0.1 * tolerance    # the lag explains it
    end
end

@testset "ForwardDiff through both demodulators" begin
    line = AnalyticLorentzLine(WMS_NU0, 2e-21, WMS_GAMMA / 1013.25)
    sigma_fine(grid) = cross_section(line, grid, GasConditions(1013.25, 296.0, 0.0))
    grid = range(WMS_NU0 - 0.4, WMS_NU0 + 0.4; step = WMS_GAMMA / 50)
    sigma = sigma_fine(grid)
    x0 = ComponentVector(CH4 = 1e19, instrument = (amplitude_cm = 0.1, linear_amplitude = 0.1,
                                                   quadratic_amplitude = 0.02,
                                                   projection_phase_rad = 0.3))
    laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1, linear_phase_rad = 0.2,
                                quadratic_amplitude = 0.02, quadratic_phase_rad = -0.4)
    f = 2e3
    cases = (
        (HarmonicExpansion(), LinearRamp(WMS_NU0 - 0.15, 0.3), collect(range(0.0, 1.0; length = 9))),
        (DigitalLockIn(lowpass = Butterworth(order = 4, cutoff_Hz = 200.0), sampling_rate_Hz = 20e3,
                       pre_roll_s = 0.02), LinearRamp(WMS_NU0 - 0.15, 6.0), collect(range(0.0, 0.05; length = 6))),
    )
    for (demodulator, scan, times) in cases, observable in (HarmonicProjection(2, 0.0, 0),
                                                           HarmonicRatio(2, 1))
        instrument = WavelengthModulation(LaserTuning(scan, SinusoidalModulation(0.05, f)), laser,
                                          demodulator, observable)
        bound = SpectralFits.CoreLayer.bind_parameters(instrument, x0)
        @test bound.tuning.modulation.amplitude_cm isa FromState
        p = prepare(bound, wms_measurement(times))
        model(v) = (x = ComponentVector(v, getaxes(x0));
                    apply_instrument(p, grid, exp.(-x.CH4 .* sigma), x))
        v0 = collect(x0)
        @inferred apply_instrument(p, grid, exp.(-x0.CH4 .* sigma), x0)
        J = ForwardDiff.jacobian(model, v0)
        Jfd = finite_difference_jacobian(model, v0)
        used = observable isa HarmonicProjection ? (1:5) : (1:4)
        for j in used
            @test norm(J[:, j] .- Jfd[:, j]) <= 1e-4 * norm(Jfd[:, j])
            @test norm(Jfd[:, j]) > 0
        end
        observable isa HarmonicRatio && @test all(iszero, J[:, 5])
    end
end
