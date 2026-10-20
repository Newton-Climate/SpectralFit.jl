# WMS observables: normalization invariance, projections and background subtraction.

function wms_obs_setup()
    nu0, gamma = 6046.95, 0.05
    line = AnalyticLorentzLine(nu0, 2e-21, gamma / 1013.25)
    grid = range(nu0 - 0.5, nu0 + 0.5; step = gamma / 100)
    transmission = exp.(-1e19 .* cross_section(line, grid, GasConditions(1013.25, 296.0, 0.0)))
    times = collect(range(0.0, 1.0; length = 31))
    tuning = LaserTuning(LinearRamp(nu0 - 0.2, 0.4), SinusoidalModulation(0.1, 10e3))
    return grid, transmission, times, tuning
end
wms_obs_measurement(times, nblocks = 1) =
    Measurement(y = zeros(nblocks * length(times)), coordinate = repeat(times, nblocks), noise_variance = 1.0)

@testset "WMS observables: gain and power invariance" begin
    grid, transmission, times, tuning = wms_obs_setup()
    for observable in (HarmonicRatio(2, 1), HarmonicOverDC(2))
        model_at(power, gain) = apply_instrument(
            prepare(WavelengthModulation(tuning,
                                         IntensityModulation(mean = power, linear_amplitude = 0.15,
                                                             linear_phase_rad = 0.4, quadratic_amplitude = 0.01),
                                         HarmonicExpansion(), observable; detector_gain = gain),
                    wms_obs_measurement(times)), grid, transmission, nothing)
        reference = model_at(1.0, 1.0)
        @test all(isfinite, reference) && maximum(abs, reference) > 1e-3
        @test model_at(3.7, 1.0) ≈ reference rtol = 1e-12
        @test model_at(1.0, 0.02) ≈ reference rtol = 1e-12
        @test model_at(0.4, 250.0) ≈ reference rtol = 1e-12
    end
    # an additive detector offset changes HarmonicOverDC but not HarmonicRatio
    base = WavelengthModulation(tuning, IntensityModulation(linear_amplitude = 0.1), HarmonicExpansion(),
                                HarmonicRatio(2, 1))
    offset = WavelengthModulation(tuning, IntensityModulation(linear_amplitude = 0.1), HarmonicExpansion(),
                                  HarmonicRatio(2, 1); detector_offset = 0.3)
    m = wms_obs_measurement(times)
    @test apply_instrument(prepare(offset, m), grid, transmission, nothing) ≈
          apply_instrument(prepare(base, m), grid, transmission, nothing) rtol = 1e-12
end

@testset "WMS observables: projection and ratios" begin
    grid, transmission, times, tuning = wms_obs_setup()
    laser = IntensityModulation(linear_amplitude = 0.1, linear_phase_rad = 0.7)
    m = wms_obs_measurement(times)
    p = prepare(WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (1, 2, 3)),
                                     HarmonicRatio(3, 1)), m)
    h = harmonic_signals(p, grid, transmission, nothing)
    X2, Y2 = in_phase(h, 2), quadrature(h, 2)
    for (phi, ref) in ((0.0, 0), (0.9, 1), (-2.0, 3))
        obs = HarmonicProjection(2, phi, ref)
        norm_ = ref == 0 ? dc_level(h) : magnitude(h, ref)
        @test evaluate_observable(obs, h, nothing) ≈ (X2 .* cos(phi) .+ Y2 .* sin(phi)) ./ norm_ rtol = 1e-14
    end
    @test evaluate_observable(HarmonicRatio(2, 0), h, nothing) ≈ evaluate_observable(HarmonicOverDC(2), h, nothing)
    @test evaluate_observable(HarmonicRatio(3, 1), h, nothing) ≈ magnitude(h, 3) ./ magnitude(h, 1)
    @test apply_instrument(p, grid, transmission, nothing) ≈ magnitude(h, 3) ./ magnitude(h, 1)
    q = evaluate_observable(Quadratures((2, 1)), h, nothing)
    @test q ≈ vcat(X2, Y2, in_phase(h, 1), quadrature(h, 1))
    @test_throws ArgumentError HarmonicRatio(2, 2)
    @test_throws ArgumentError WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (1, 2)),
                                                    HarmonicRatio(3, 1))
    @test_throws ArgumentError magnitude(h, 4)
    # projection phase slot binds to (:instrument, :projection_phase_rad)
    x = ComponentVector(instrument = (projection_phase_rad = 0.25,))
    bound = SpectralFits.CoreLayer.bind_parameters(HarmonicProjection(2, 0.0, 1), x)
    @test bound.phase_rad isa FromState{(:instrument, :projection_phase_rad)}
    @test evaluate_observable(bound, h, x) ≈ evaluate_observable(HarmonicProjection(2, 0.25, 1), h, nothing)
    # Quadratures measurements must repeat the times in every block
    bad = Measurement(y = zeros(4), coordinate = [0.0, 1.0, 0.0, 2.0], noise_variance = 1.0)
    @test_throws ArgumentError prepare(WavelengthModulation(tuning, laser, HarmonicExpansion(),
                                                            Quadratures((1,))), bad)
end

@testset "WMS observables: background" begin
    grid, transmission, times, tuning = wms_obs_setup()
    ones_T = ones(length(grid))
    m2 = wms_obs_measurement(times, 2)
    I0 = 1.3
    # transmission 1, purely linear intensity modulation: no second harmonic
    linear = IntensityModulation(mean = I0, linear_amplitude = 0.2, linear_phase_rad = 0.5)
    p = prepare(WavelengthModulation(tuning, linear, HarmonicExpansion(), Quadratures((2,))), m2)
    @test maximum(abs, apply_instrument(p, grid, ones_T, nothing)) < 1e-14 * I0
    # lock-in: exact zero up to the filter's ripple rejection, (100 Hz / 10 kHz)^5 ~ 1e-10,
    # once the 0.2 s pre-roll has let the filter settle
    lockin = DigitalLockIn(sampling_rate_Hz = 200e3, pre_roll_s = 0.2)
    p = prepare(WavelengthModulation(tuning, linear, lockin, Quadratures((2,))),
                wms_obs_measurement(collect(range(0.0, 0.004; length = 5)), 2))
    @test maximum(abs, apply_instrument(p, grid, ones_T, nothing)) < 1e-9 * I0

    # quadratic intensity modulation: X2 = I0 i2 cos(psi2), Y2 = -I0 i2 sin(psi2)
    i2, psi2 = 0.03, 0.6
    quad = IntensityModulation(mean = I0, linear_amplitude = 0.2, linear_phase_rad = 0.5,
                               quadratic_amplitude = i2, quadratic_phase_rad = psi2)
    p = prepare(WavelengthModulation(tuning, quad, HarmonicExpansion(), Quadratures((2,))), m2)
    y = apply_instrument(p, grid, ones_T, nothing)
    @test y[1:31] ≈ fill(I0 * i2 * cos(psi2), 31) rtol = 1e-12
    @test y[32:62] ≈ fill(-I0 * i2 * sin(psi2), 31) rtol = 1e-12

    for inner in (Quadratures((2,)), HarmonicOverDC(2), HarmonicProjection(2, 0.3, 0))
        mm = inner isa Quadratures ? m2 : wms_obs_measurement(times)
        instrument = WavelengthModulation(tuning, quad, HarmonicExpansion(), BackgroundSubtracted(inner))
        pb = prepare(instrument, mm)
        @test pb.background isa Vector{Float64}                       # cached at prepare
        @test maximum(abs, apply_instrument(pb, grid, ones_T, nothing)) < 1e-14
        with_line = apply_instrument(pb, grid, transmission, nothing)
        plain = apply_instrument(prepare(WavelengthModulation(tuning, quad, HarmonicExpansion(), inner), mm),
                                 grid, transmission, nothing)
        background = apply_instrument(prepare(WavelengthModulation(tuning, quad, HarmonicExpansion(), inner), mm),
                                      grid, ones_T, nothing)
        @test with_line ≈ plain .- background rtol = 1e-12
        # retrieved laser parameters: the background is recomputed from the state at each call
        x = ComponentVector(instrument = (quadratic_amplitude = i2,))
        bound = SpectralFits.CoreLayer.bind_parameters(instrument, x)
        pd = prepare(bound, mm)
        @test pd.background === nothing
        @test apply_instrument(pd, grid, transmission, x) ≈ with_line rtol = 1e-12
        # binding after prepare gives the same result
        pd2 = SpectralFits.CoreLayer.bind_parameters(pb, x)
        @test pd2.background === nothing
        @test apply_instrument(pd2, grid, transmission, x) ≈ with_line rtol = 1e-12
    end
end
