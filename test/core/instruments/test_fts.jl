# FTS instrument line shape and sparse ILS matrix.

fts_measurement(nu) = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1.0)

@testset "FTS line shape" begin
    L = 1.0
    fts = FTS(max_path_difference_cm = L)
    @test instrument_line_shape(fts, 0.0) ≈ 2L

    # integral over a symmetric window ending half-way between sinc zeros (K = 1000.5 lobes):
    # the truncation error is (2/pi^3)/K^2 < 1e-7, so the full integral is 1 to 1e-6
    K = 1000.5
    h = 1 / (2L) / 40
    delta = range(-K / (2L), K / (2L); step = h)
    vals = instrument_line_shape.(Ref(fts), delta)
    integral = h * (sum(vals) - (vals[1] + vals[end]) / 2)
    @test integral ≈ 1 rtol = 1e-6

    tri = FTS(max_path_difference_cm = L, apodization = Triangular())
    vals_tri = instrument_line_shape.(Ref(tri), delta)
    @test h * (sum(vals_tri) - (vals_tri[1] + vals_tri[end]) / 2) ≈ 1 rtol = 1e-3   # 1/delta^2 tail

    # full width at half maximum by bisection
    function fwhm(instr)
        peak = instrument_line_shape(instr, 0.0)
        lo, hi = 0.0, 1 / instr.max_path_difference_cm
        for _ in 1:200
            mid = (lo + hi) / 2
            instrument_line_shape(instr, mid) > peak / 2 ? (lo = mid) : (hi = mid)
        end
        return 2lo
    end
    for Lx in (0.5, 1.0, 45.0)
        @test fwhm(FTS(max_path_difference_cm = Lx)) ≈ 1.207 / (2Lx) rtol = 1e-3
        @test fwhm(FTS(max_path_difference_cm = Lx, apodization = Triangular())) ≈ 1.7718 / (2Lx) rtol = 1e-3
    end
end

@testset "FTS matrix" begin
    L = 1.0
    fts = FTS(max_path_difference_cm = L)
    nu0 = 1000.0
    observed = collect(range(nu0 - 3.0, nu0 + 3.0; length = 241))
    p = prepare(fts, fts_measurement(observed))
    @test n_observations(p) == 241
    lo, hi = wavenumber_coverage_cm(p, nothing)
    @test lo ≈ nu0 - 3.0 - fts.ils_half_width_cm
    @test hi ≈ nu0 + 3.0 + fts.ils_half_width_cm
    grid = default_fine_grid(p; oversampling = 10)
    @test first(grid) <= lo && last(grid) >= hi
    pb = bind_fine_grid(p, grid)
    @test size(pb.matrix) == (241, length(grid))
    @test all(sum(pb.matrix; dims = 2) .≈ 1)

    # monochromatic line: a unit-area spike on the fine grid returns the line shape
    j0 = argmin(abs.(grid .- nu0))
    spike = zeros(length(grid)); spike[j0] = 1 / step(grid)
    y = @inferred apply_instrument(pb, grid, spike, nothing)
    expected = instrument_line_shape.(Ref(fts), grid[j0] .- observed)
    @test maximum(abs, y .- expected) < 1e-4 * 2L

    # unbound prepared instruments build the matrix on the fly; grid mismatch is caught
    @test apply_instrument(p, grid, spike, nothing) ≈ y
    @test_throws DimensionMismatch apply_instrument(pb, grid, spike[1:end-1], nothing)
    @test_throws ArgumentError bind_fine_grid(p, range(nu0 - 1, nu0 + 1; step = 0.05))

    # constant transmission is preserved
    @test apply_instrument(pb, grid, fill(0.7, length(grid)), nothing) ≈ fill(0.7, 241) rtol = 1e-12

    # field of view shifts a line down by nu theta^2 / 4
    theta = 0.01
    fov = FTS(max_path_difference_cm = L, apodization = Triangular(), field_of_view_rad = theta)
    pf = prepare(fov, fts_measurement(observed))
    gf = default_fine_grid(pf; oversampling = 10)
    pfb = bind_fine_grid(pf, gf)
    j0 = argmin(abs.(gf .- nu0))
    spike = zeros(length(gf)); spike[j0] = 1 / step(gf)
    shift = gf[j0] * theta^2 / 4
    obs_fine = collect(range(gf[j0] - shift - 2.0, gf[j0] - shift + 2.0; length = 2001))
    yf = apply_instrument(bind_fine_grid(prepare(fov, fts_measurement(obs_fine)), gf), gf, spike, nothing)
    centroid = sum(yf .* obs_fine) / sum(yf)
    @test centroid ≈ gf[j0] - shift atol = 1e-3 * shift
    @test wavenumber_coverage_cm(pf, nothing)[2] > hi

    # dual numbers: Jacobian of the convolved Beer-Lambert spectrum against a column
    line = AnalyticLorentzLine(nu0, 1e-20, 0.1 / 1013.25)
    sigma = cross_section(line, grid, GasConditions(1013.25, 296.0, 0.0))
    f(c) = apply_instrument(pb, grid, exp.(-c[1] .* sigma), nothing)
    c0 = [3e18]
    @test ForwardDiff.jacobian(f, c0) ≈ finite_difference_jacobian(f, c0) rtol = 1e-6
end
