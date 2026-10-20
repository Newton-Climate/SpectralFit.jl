# Grating spectrometer: a tabulated Gaussian line shape applied to a Gaussian absorption line
# reproduces the analytic convolution, 1 - A (s_l / s_t) exp(-(nu - nu0)^2 / (2 s_t^2)) with
# s_t^2 = s_l^2 + s_i^2.

@testset "Grating spectrometer" begin
    nu0 = 6000.0
    A = 0.3
    s_line = 0.03
    gaussian(d, s) = exp(-d^2 / (2s^2))
    convolved(nu, s_ils) = 1 - A * s_line / hypot(s_line, s_ils) * gaussian(nu - nu0, hypot(s_line, s_ils))
    pixels = collect(range(nu0 - 0.4, nu0 + 0.4; length = 57))
    meas = Measurement(y = ones(57), coordinate = pixels, noise_variance = 1.0)

    s_ils = 0.05
    offsets = range(-8s_ils, 8s_ils; length = 801)
    g = GratingSpectrometer(offsets_cm = offsets, response = gaussian.(offsets, s_ils))
    p = prepare(g, meas)
    @test n_observations(p) == 57
    @test all(wavenumber_coverage_cm(p, nothing) .≈ (nu0 - 0.4 - 8s_ils, nu0 + 0.4 + 8s_ils))
    grid = range(nu0 - 1.0, nu0 + 1.0; step = 0.001)
    transmission = 1 .- A .* gaussian.(grid .- nu0, s_line)
    pb = bind_fine_grid(p, grid)
    y = @inferred apply_instrument(pb, grid, transmission, nothing)
    expected = convolved.(pixels, s_ils)
    @test maximum(abs, y .- expected) / A < 1e-6
    @test apply_instrument(p, grid, transmission, nothing) ≈ y          # built on the fly

    # one line shape per pixel, widths varying across the detector
    widths = range(0.03, 0.07; length = 57)
    table = hcat([gaussian.(offsets, w) for w in widths]...)
    gp = GratingSpectrometer(offsets_cm = offsets, response = table)
    yp = apply_instrument(bind_fine_grid(prepare(gp, meas), grid), grid, transmission, nothing)
    @test maximum(abs, yp .- convolved.(pixels, widths)) / A < 1e-6

    # irregular table: linear interpolation, looser
    irregular = sort(vcat(collect(offsets[1:2:end]), [0.0123, -0.0311]))
    gl = GratingSpectrometer(offsets_cm = irregular, response = gaussian.(irregular, s_ils))
    yl = apply_instrument(bind_fine_grid(prepare(gl, meas), grid), grid, transmission, nothing)
    @test maximum(abs, yl .- expected) / A < 1e-4

    @test_throws DimensionMismatch prepare(GratingSpectrometer(offsets_cm = offsets, response = table[:, 1:3]), meas)
    @test_throws ArgumentError GratingSpectrometer(offsets_cm = reverse(collect(offsets)),
                                                   response = gaussian.(offsets, s_ils))
    dg = default_fine_grid(p)
    @test step(dg) ≈ step(offsets)
    @test first(dg) <= wavenumber_coverage_cm(p, nothing)[1] && last(dg) >= wavenumber_coverage_cm(p, nothing)[2]
end
