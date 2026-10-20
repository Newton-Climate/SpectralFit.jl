# Phase 8: noise estimation and noise models.

@testset "estimate_noise_variance recovers white noise" begin
    rng = MersenneTwister(20160921)
    nu = collect(range(6076.0, 6078.0; length = 10_000))
    sigma2 = 2.5e-5
    baseline = @. 1.0 + 0.2 * (nu - 6077) - 0.3 * (nu - 6077)^2 + 0.1 * (nu - 6077)^3
    y = baseline .+ sqrt(sigma2) .* randn(rng, length(nu))
    # one window covering everything, and two windows: both within 5 percent
    est_all = estimate_noise_variance(nu, y; flat_windows_cm = [(6075.9, 6078.1)],
                                      polynomial_order = 3)
    @test isapprox(est_all, sigma2; rtol = 0.05)
    est_two = estimate_noise_variance(nu, y; flat_windows_cm = [(6076.0, 6076.9), (6077.1, 6078.0)])
    @test isapprox(est_two, sigma2; rtol = 0.05)
    # matrix form: one value per record
    Y = vcat(y', (2 .* baseline .+ 2 * sqrt(sigma2) .* randn(rng, length(nu)))')
    est = estimate_noise_variance(nu, Y; flat_windows_cm = [(6075.9, 6078.1)])
    @test length(est) == 2
    @test isapprox(est[2], 4 * sigma2; rtol = 0.05)
    # a noise-free polynomial gives (numerically) zero
    @test estimate_noise_variance(nu, baseline; flat_windows_cm = [(6076.0, 6078.0)]) < 1e-20
    @test_throws ArgumentError estimate_noise_variance(nu, y; flat_windows_cm = [(7000.0, 7001.0)])
end

@testset "shot_noise_variance scales with sqrt(signal)" begin
    rng = MersenneTwister(7)
    nu = collect(range(6255.0, 6255.5; length = 400))
    levels = repeat([1.0, 4.0, 9.0, 16.0], 10)   # 40 records: the across-record
    # mean removal biases the variance low by a factor (n - 1) / n, as in v0.1
    a = 1e-3
    Y = reduce(vcat, ((level .+ a * sqrt(level) .* randn(rng, length(nu)))' for level in levels))
    v = shot_noise_variance(nu, Y)
    @test length(v) == 40
    @test all(isapprox.(v, (a .* sqrt.(levels)) .^ 2; rtol = 0.15))
    @test v[4] / v[1] ≈ 16.0 rtol = 1e-3                       # model scaling
    @test_throws ArgumentError shot_noise_variance(nu, Y[1:1, :])
end

@testset "noise models and with_noise_variance" begin
    rng = MersenneTwister(3)
    nu = collect(range(6076.0, 6078.0; length = 2000))
    ms = [Measurement(y = 1.0 .+ 0.01 .* randn(rng, length(nu)), coordinate = nu,
                      noise_variance = 1.0, time = DateTime(2020, 1, 1) + Second(i),
                      conditions = (pressure_hPa = 850.0,), metadata = (n_averaged = 1,))
          for i in 1:3]
    model = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)],
                          polynomial_order = 3)
    m2 = with_noise_variance(ms[1], model)
    @test m2.y == ms[1].y && m2.conditions == ms[1].conditions && m2.time == ms[1].time
    @test all(==(m2.noise_variance[1]), m2.noise_variance)
    @test isapprox(m2.noise_variance[1], 1e-4; rtol = 0.15)
    @test ms[1].noise_variance == ones(length(nu))              # input unchanged
    ms3 = with_noise_variance(ms, ConstantNoise(2e-4))
    @test all(m -> m.noise_variance == fill(2e-4, length(nu)), ms3)
    @test length(with_noise_variance(ms, ShotNoise(flat_window_cm = (6076.1, 6076.6)))) == 3
    @test_throws ArgumentError with_noise_variance(ms[1], ShotNoise())
    @test_throws ArgumentError ConstantNoise(0.0)
    @test_throws ArgumentError ResidualNoise(flat_windows_cm = [(2.0, 1.0)])
    @test record_noise_variances(ConstantNoise(3.0), nu, zeros(2, length(nu))) == [3.0, 3.0]
end
