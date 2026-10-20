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
