# WMS laser tuning and intensity against their closed forms.

@testset "WMS tuning" begin
    t = collect(range(0.0, 0.25; length = 1001))
    a, f, phi = 0.05, 1.3e3, 0.4
    mod = SinusoidalModulation(a, f, phi)
    @test mod === SinusoidalModulation(amplitude_cm = a, frequency_Hz = f, phase_rad = phi)
    m(ti) = a * cos(2pi * f * ti + phi)

    tuning = LaserTuning(FixedCenter(6046.95), mod)
    @test wavenumber_cm.(Ref(tuning), t, nothing) ≈ 6046.95 .+ m.(t) rtol = 1e-14

    tuning = LaserTuning(LinearRamp(start_cm = 6046.5, rate_cm_per_s = 2.0), mod)
    @test wavenumber_cm.(Ref(tuning), t, nothing) ≈ 6046.5 .+ 2.0 .* t .+ m.(t) rtol = 1e-14

    start, span, period = 6046.3, 0.8, 0.1
    tuning = LaserTuning(TriangleScan(start_cm = start, span_cm = span, period_s = period), mod)
    function triangle(ti)
        u = mod1(ti, period) == period ? 0.0 : rem(ti, period) / period
        return u < 0.5 ? start + 2span * u : start + 2span * (1 - u)
    end
    @test wavenumber_cm.(Ref(tuning), t, nothing) ≈ triangle.(t) .+ m.(t) rtol = 1e-14
    @test wavenumber_cm(tuning, period / 2, nothing) ≈ start + span + m(period / 2) rtol = 1e-14

    # slots read from the state after binding; default paths under :instrument
    x = ComponentVector(instrument = (start_cm = 6046.0, span_cm = 1.0, period_s = 0.2,
                                      amplitude_cm = 0.07, modulation_phase_rad = 0.1))
    bound = SpectralFits.CoreLayer.bind_parameters(tuning, x)
    @test bound.scan.span_cm isa FromState{(:instrument, :span_cm)}
    @test bound.modulation.phase_rad isa FromState{(:instrument, :modulation_phase_rad)}
    @test wavenumber_cm(bound, 0.05, x) ≈ 6046.0 + 0.5 + 0.07 * cos(2pi * f * 0.05 + 0.1) rtol = 1e-14

    # dual numbers through every slot, including the triangle period
    g(v) = wavenumber_cm(bound, 0.137, ComponentVector(v, getaxes(x)))
    @test ForwardDiff.gradient(g, collect(x)) ≈ finite_difference_jacobian(v -> [g(v)], collect(x))[1, :] rtol = 1e-6
end

@testset "WMS intensity" begin
    t = collect(range(0.0, 0.01; length = 777))
    f, phi = 2.1e3, -0.3
    mod = SinusoidalModulation(0.05, f, phi)
    I0, i1, psi1, i2, psi2, slope = 1.7, 0.12, 0.8, 0.03, -1.1, 0.5
    laser = IntensityModulation(mean = I0, linear_amplitude = i1, linear_phase_rad = psi1,
                                quadratic_amplitude = i2, quadratic_phase_rad = psi2,
                                scan_slope = slope, modulation = mod)
    th(ti) = 2pi * f * ti + phi
    closed(ti) = I0 * (1 + slope * ti) * (1 + i1 * cos(th(ti) + psi1) + i2 * cos(2th(ti) + psi2))
    @test laser_intensity.(Ref(laser), t, nothing) ≈ closed.(t) rtol = 1e-14

    simple = IntensityModulation(mean = 2.0, linear_amplitude = 0.1, modulation = SinusoidalModulation(0.05, 1e3))
    @test laser_intensity(simple, 0.0, nothing) ≈ 2.2 rtol = 1e-14
    @test_throws ArgumentError laser_intensity(IntensityModulation(mean = 1.0), 0.0, nothing)

    x = ComponentVector(instrument = (mean = 3.0, linear_amplitude = 0.2, quadratic_amplitude = 0.05))
    bound = SpectralFits.CoreLayer.bind_parameters(laser, x)
    @test bound.mean isa FromState{(:instrument, :mean)}
    @test bound.linear_phase_rad isa Fixed
    @test laser_intensity(bound, 0.003, x) ≈
          3.0 * (1 + slope * 0.003) * (1 + 0.2 * cos(th(0.003) + psi1) + 0.05 * cos(2th(0.003) + psi2)) rtol = 1e-14
end
