@testset "Measurement construction and select_window" begin
    nu = collect(6000.0:0.5:6005.0)
    m = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1e-4,
                    time = DateTime(2020, 1, 1), conditions = (pressure_hPa = 850.0,))
    @test m.noise_variance == fill(1e-4, length(nu))
    @test m.conditions.pressure_hPa == 850.0
    w = select_window(m, 6001.0, 6003.0)
    @test w.coordinate == [6001.0, 6001.5, 6002.0, 6002.5, 6003.0]     # closed interval
    @test w.time == m.time
    @test_throws DimensionMismatch Measurement(y = ones(3), coordinate = [1.0, 2.0], noise_variance = 1.0)
    @test_throws ArgumentError Measurement(y = ones(2), coordinate = [1.0, 2.0], noise_variance = 0.0)
    @test_throws ArgumentError select_window(m, 7000.0, 7001.0)
end

@testset "fine-grid interpolation" begin
    grid = 6000.0:0.01:6001.0
    f(nu) = 1 + 0.3 * (nu - 6000.5) - 2.0 * (nu - 6000.5)^2 + 5.0 * (nu - 6000.5)^3
    q = [6000.733, 6000.2, 6000.51, 6000.05]          # unsorted
    v = interpolate_fine_grid(grid, f.(grid), q)
    @test all(isapprox.(v, f.(q); rtol = 1e-6))
end
