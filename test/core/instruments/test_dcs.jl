# Tests for src/core/instruments/dcs.jl: interpolation onto measured wavenumbers, stretch
# calibration, coverage, fine grid, binding and type stability.

dcs_dual_state(x) = ComponentVector(ForwardDiff.Dual{Nothing}.(collect(x), 1.0), getaxes(x))
dcs_cubic(nu) = 0.3 + 0.02 * (nu - 6077.0) - 0.5 * (nu - 6077.0)^2 + 0.7 * (nu - 6077.0)^3

function dcs_test_measurement(nu)
    return Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1e-4)
end

@testset "DCS reproduces a cubic polynomial, sorted and unsorted" begin
    fine = range(6075.0, 6079.0; step = 0.002)
    values = dcs_cubic.(fine)
    rng = MersenneTwister(7)
    nu_sorted = collect(range(6076.0, 6078.0; length = 301))
    nu_unsorted = 6076.0 .+ 2.0 .* rand(rng, 300)
    for nu in (nu_sorted, nu_unsorted)
        prepared = prepare(DCS(), dcs_test_measurement(nu))
        @test prepared isa PreparedDCS
        @test n_observations(prepared) == length(nu)
        @test prepared.wavenumber_cm == nu
        out = apply_instrument(prepared, fine, values, ComponentVector(CH4 = 1.0))
        @test maximum(abs.(out .- dcs_cubic.(nu))) < 1e-10
        @test out == interpolate_fine_grid(fine, values, nu)
    end
    # unsorted queries give the sorted answer permuted
    prepared = prepare(DCS(), dcs_test_measurement(nu_unsorted))
    order = sortperm(nu_unsorted)
    sorted_out = apply_instrument(prepare(DCS(), dcs_test_measurement(nu_unsorted[order])),
                                  fine, values, nothing)
    @test apply_instrument(prepared, fine, values, nothing)[order] == sorted_out
end

@testset "LinearStretch" begin
    fine = range(6075.0, 6079.0; step = 0.002)
    values = dcs_cubic.(fine)
    nu = collect(range(6076.0, 6078.0; length = 101))
    m = dcs_test_measurement(nu)
    x = ComponentVector(CH4 = 1.0, instrument = (stretch = 1.0,))
    plain = apply_instrument(prepare(DCS(), m), fine, values, x)
    @test apply_instrument(prepare(DCS(calibration = LinearStretch()), m), fine, values, x) == plain
    @test apply_instrument(prepare(DCS(calibration = LinearStretch(stretch = 1.0)), m), fine, values,
                           ComponentVector(CH4 = 1.0)) == plain
    # stretch read from the state: observed wavenumber = stretch * model wavenumber
    instrument = bind_parameters(DCS(calibration = LinearStretch()), x)
    @test instrument.calibration.stretch isa FromState{(:instrument, :stretch)}
    prepared = prepare(instrument, m)
    @test apply_instrument(prepared, fine, values, x) == plain
    s = 1 + 1e-4
    xs = ComponentVector(CH4 = 1.0, instrument = (stretch = s,))
    @test maximum(abs.(apply_instrument(prepared, fine, values, xs) .- dcs_cubic.(nu ./ s))) < 1e-10
    lo, hi = wavenumber_coverage_cm(prepared, xs)
    @test lo == minimum(nu) / s && hi == maximum(nu) / s
    # no stretch entry in the state: the slot stays fixed
    @test bind_parameters(DCS(calibration = LinearStretch()), ComponentVector(CH4 = 1.0)).calibration.stretch isa Fixed
    @test bind_parameters(DCS(), x) == DCS()
    @test bind_parameters(prepare(DCS(calibration = LinearStretch()), m), x).calibration.stretch isa FromState
    @test_throws ArgumentError bind_parameters(
        DCS(calibration = LinearStretch(stretch = FromState{(:instrument, :stretch)}())), ComponentVector(CH4 = 1.0))
end

@testset "coverage and default fine grid" begin
    nu = collect(range(6076.05, 6077.95; length = 191))   # spacing 0.01
    prepared = prepare(DCS(), dcs_test_measurement(reverse(nu)))
    @test wavenumber_coverage_cm(prepared, nothing) == (6076.05, 6077.95)
    grid = default_fine_grid(prepared)
    @test isapprox(step(grid), 0.005; rtol = 1e-9)
    @test first(grid) ≈ 6075.05 && last(grid) >= 6078.95 - 1e-9
    grid2 = default_fine_grid(prepared; spacing_cm = 0.003, margin_cm = 0.5)
    @test step(grid2) == 0.003
    @test first(grid2) ≈ 6075.55 && last(grid2) >= 6078.45 - 1e-9
    @test isapprox(step(default_fine_grid(prepared; oversampling = 4)), 0.0025; rtol = 1e-9)
    stretched = prepare(bind_parameters(DCS(calibration = LinearStretch()),
                                        ComponentVector(instrument = (stretch = 1.0,))),
                        dcs_test_measurement(nu))
    @test first(default_fine_grid(stretched)) ≈ 6075.05
    @test first(default_fine_grid(stretched; x_prior = ComponentVector(instrument = (stretch = 2.0,)))) ≈
          6076.05 / 2 - 1
    @test_throws ArgumentError default_fine_grid(prepare(DCS(), dcs_test_measurement([6076.0])))
    @test_throws ArgumentError default_fine_grid(prepared; spacing_cm = -1.0)
end

@testset "type stability and dual numbers in apply_instrument" begin
    fine = range(6075.0, 6079.0; step = 0.002)
    values = dcs_cubic.(fine)
    nu = collect(range(6076.0, 6078.0; length = 101))
    m = dcs_test_measurement(nu)
    x = ComponentVector(CH4 = 1.0, instrument = (stretch = 1.0,))
    xd = dcs_dual_state(x)
    plain = prepare(DCS(), m)
    stretched = prepare(bind_parameters(DCS(calibration = LinearStretch()), x), m)
    for prepared in (plain, stretched)
        @test (@inferred apply_instrument(prepared, fine, values, x)) isa Vector{Float64}
        dual_values = ForwardDiff.Dual{Nothing}.(values, 1.0)
        @test (@inferred apply_instrument(prepared, fine, dual_values, x)) isa
              Vector{ForwardDiff.Dual{Nothing,Float64,1}}
        expected = prepared === plain ? Float64 : eltype(xd)   # no stretch: x is not read
        @test (@inferred apply_instrument(prepared, fine, values, xd)) isa Vector{expected}
    end
    # derivative with respect to the stretch: d/ds f(nu / s) = -nu / s^2 * f'(nu / s)
    f = v -> apply_instrument(stretched, fine, values, ComponentVector(v, getaxes(x)))
    J = ForwardDiff.jacobian(f, collect(x))
    dcubic(n) = 0.02 - 1.0 * (n - 6077.0) + 2.1 * (n - 6077.0)^2
    @test maximum(abs.(J[:, 2] .- (-nu .* dcubic.(nu)))) < 1e-6 * maximum(abs.(nu .* dcubic.(nu)))
    @test all(iszero, J[:, 1])
end

@testset "sparse interpolation matrix after bind_fine_grid" begin
    fine = range(6075.0, 6079.0; step = 0.002)
    rng = MersenneTwister(11)
    values = 1 .- 0.4 .* exp.(-((fine .- 6077.0) ./ 0.01) .^ 2) .+ 0.01 .* randn(rng, length(fine))
    # unsorted interior points, both grid ends, points next to the ends, grid nodes
    nu = vcat(6075.0 .+ 4.0 .* rand(rng, 400), first(fine), last(fine), first(fine) + 1e-6,
              last(fine) - 1e-6, fine[2] + 0.3 * step(fine), collect(fine[1:7:end]))
    m = dcs_test_measurement(nu)
    x = ComponentVector(CH4 = 1.0)
    for calibration in (NoCalibration(), LinearStretch(stretch = 1.0), LinearStretch(stretch = 1 + 1e-5))
        query = calibration isa NoCalibration ? nu : nu ./ resolve(calibration.stretch, nothing)
        keep = first(fine) .<= query .<= last(fine)   # a stretch moves some points off the grid
        unbound = prepare(DCS(calibration = calibration), dcs_test_measurement(nu[keep]))
        bound = bind_fine_grid(unbound, fine)
        @test bound.matrix isa SparseMatrixCSC{Float64,Int}
        @test bound.wavenumber_fine_cm === fine
        spline = interpolate_fine_grid(fine, values, query[keep])
        @test maximum(abs.(apply_instrument(bound, fine, values, x) .- spline)) < 1e-12
        @test apply_instrument(unbound, fine, values, x) == spline   # unbound: B-spline path
        for _ in 1:3   # the matrix is the linear map, not a fit to one spectrum
            v = randn(rng, length(fine))
            @test maximum(abs.(bound.matrix * v .- interpolate_fine_grid(fine, v, query[keep]))) < 1e-12
        end
        dual_values = ForwardDiff.Dual{Nothing}.(values, 1.0)
        @test (@inferred apply_instrument(bound, fine, dual_values, x)) isa
              Vector{ForwardDiff.Dual{Nothing,Float64,1}}
        @test (@inferred apply_instrument(bound, fine, values, x)) isa Vector{Float64}
    end
    # the bound grid is checked; queries outside the grid are refused
    bound = bind_fine_grid(prepare(DCS(), m), fine)
    @test_throws ArgumentError apply_instrument(bound, range(6075.0, 6079.0; step = 0.004),
                                                values[1:2:end], x)
    @test_throws DimensionMismatch apply_instrument(bound, fine, values[2:end], x)
    @test_throws ArgumentError bind_fine_grid(prepare(DCS(), dcs_test_measurement([6074.9])), fine)
    # rebinding the parameters keeps a matrix consistent with the calibration
    rebound = bind_parameters(bind_fine_grid(prepare(DCS(calibration = LinearStretch()), m), fine),
                              ComponentVector(instrument = (stretch = 1.0,)))
    @test rebound.calibration.stretch isa FromState && rebound.matrix === nothing
end

@testset "stretch from the state keeps the spline; Jacobian matches finite differences" begin
    fine = range(6075.0, 6079.0; step = 0.002)
    values = 1 .- 0.4 .* exp.(-((fine .- 6077.0) ./ 0.05) .^ 2)
    nu = collect(range(6076.0, 6078.0; length = 101))[randperm(MersenneTwister(3), 101)]
    m = dcs_test_measurement(nu)
    x = ComponentVector(CH4 = 1.0, instrument = (stretch = 1.0 + 2e-5,))
    prepared = prepare(bind_parameters(DCS(calibration = LinearStretch()), x), m)
    bound = bind_fine_grid(prepared, fine)
    @test bound.matrix === nothing && bound.wavenumber_fine_cm === nothing
    @test apply_instrument(bound, fine, values, x) == interpolate_fine_grid(fine, values, nu ./ x.instrument.stretch)
    f = v -> apply_instrument(bound, fine, values, ComponentVector(v, getaxes(x)))
    J = ForwardDiff.jacobian(f, collect(x))
    h = 1e-9   # stretch step: shifts the queries by ~6e-6 cm-1
    fd = (f(collect(x) .+ [0, h]) .- f(collect(x) .- [0, h])) ./ (2h)
    @test maximum(abs.(J[:, 2] .- fd)) < 1e-5 * maximum(abs.(fd))
    @test all(iszero, J[:, 1])
end

@testset "interpolation matrix build cost on a 2e5-point grid" begin
    fine = range(6000.0; step = 0.001, length = 200_000)
    rng = MersenneTwister(5)
    nu = 6000.0 .+ 199.9 .* rand(rng, 50_000)
    prepared = prepare(DCS(), dcs_test_measurement(nu))
    bind_fine_grid(prepare(DCS(), dcs_test_measurement(nu[1:10])), fine)   # compile
    t = @elapsed bound = bind_fine_grid(prepared, fine)
    @test t < 10.0
    @test nnz(bound.matrix) < 60 * length(nu)
    v = 1 .+ 0.1 .* randn(rng, length(fine))
    @test maximum(abs.(apply_instrument(bound, fine, v, nothing) .- interpolate_fine_grid(fine, v, nu))) < 1e-12
end
