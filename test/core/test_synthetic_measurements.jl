# synthetic_measurement, apply_conditions, fine_grid_margin_cm.

@testset "synthetic_measurement adds intensity noise in either space" begin
    nu = collect(6076.0:0.01:6080.0)
    template = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1.0,
                           conditions = (pressure_hPa = 900.0,))
    signal = 0.5 .+ 0.1 .* sin.(nu)
    lin = x -> signal
    m = synthetic_measurement(lin, nothing, template, LinearSpace(), 1e-3; rng = MersenneTwister(1))
    @test m.coordinate == nu && m.conditions == template.conditions
    @test all(m.noise_variance .== 1e-6)
    @test isapprox(std(m.y .- signal), 1e-3; rtol = 0.1)
    m2 = synthetic_measurement(lin, nothing, template, LinearSpace(), 1e-3; rng = MersenneTwister(1))
    @test m2.y == m.y                                         # reproducible with the seed
    logm = synthetic_measurement(x -> log.(signal), nothing, template, LogSpace(), 1e-3; rng = MersenneTwister(1))
    @test logm.y ≈ m.y rtol = 1e-12                           # same intensity noise
    @test_throws DimensionMismatch synthetic_measurement(x -> signal[1:3], nothing, template, LinearSpace(), 1e-3)
    @test_throws r"non-positive" synthetic_measurement(x -> fill(log(1e-6), length(nu)), nothing, template,
                                                       LogSpace(), 1.0; rng = MersenneTwister(2))
end

@testset "apply_conditions uses the first sample time" begin
    t0 = DateTime(2016, 9, 21, 9, 0, 0)
    nu = [6076.0, 6076.1]
    records = [Measurement(y = [1.0, 1.0], coordinate = nu, noise_variance = 1e-4, time = t0 + Minute(14) + Second(s),
                           conditions = (pressure_hPa = 800.0, temperature_K = 290.0)) for s in (0, 30)]
    avg = only(time_average(records, Minute(15)))
    @test avg.time == t0
    @test avg.metadata.first_sample_time == t0 + Minute(14)
    table = ConditionsTable([t0, t0 + Minute(14) + Second(10)], (pressure_hPa = [700.0, 820.0], temperature_K = [280.0, 295.0]))
    m = apply_conditions(avg, table)
    @test m.conditions == (pressure_hPa = 820.0, temperature_K = 295.0)
    @test m.metadata.applied_conditions == (pressure_hPa = 820.0, temperature_K = 295.0)
    far = ConditionsTable([t0], (pressure_hPa = [700.0],); max_offset = Second(60))
    @test_throws r"no conditions within" apply_conditions(avg, far)
    @test_throws DimensionMismatch ConditionsTable([t0], (pressure_hPa = [1.0, 2.0],))
end

@testset "fine_grid_margin_cm" begin
    nu = collect(6077.0:0.0067:6079.0)
    m = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1e-6,
                    conditions = (pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5))
    absorbers = (CH4 = Absorber(:CH4, fixture_voigt(:CH4, (6076.5, 6079.5))),)
    x = ComponentVector(CH4 = 1e19, baseline = (window_1 = [1.0],))
    config(margin) = RetrievalConfig(windows_cm = [(6077.0, 6079.0)], baseline_coefficients = [1],
                                     fine_grid_spacing_cm = 0.002, fine_grid_margin_cm = margin)
    model = SF.CoreLayer.build_forward_model(config(0.1), absorbers, m, x)
    @test first(model.wavenumber_fine_cm) ≈ first(nu) - 0.1
    @test last(model.wavenumber_fine_cm) >= last(nu) + 0.1 - 1e-9
    # the default 1 cm margin leaves the absorber's range
    @test_throws r"outside the range" SF.CoreLayer.build_forward_model(config(nothing), absorbers, m, x)
    @test_throws r"fine_grid_margin_cm" config(-1.0)
end
