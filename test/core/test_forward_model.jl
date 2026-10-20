# Phase 4: forward model, builder and multi-window model.

let
    fine = 6077.0:0.01:6083.0
    nu_obs = collect(6078.0:0.0067:6082.0)
    atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
    n_air = dry_air_column(atm)
    absorbers = (H2O = Absorber(:H2O, fixture_voigt(:H2O, (6077.0, 6083.0))),
                 CH4 = Absorber(:CH4, fixture_voigt(:CH4, (6077.0, 6083.0))))
    m = Measurement(y = ones(length(nu_obs)), coordinate = nu_obs, noise_variance = 1e-8,
                    conditions = (pressure_hPa = 850.0, temperature_K = 290.0,
                                  path_length_cm = 195017.0))
    baseline = LegendreBaseline(3)
    build(x; instrument = DCS(), units = ColumnDensity(), space = LinearSpace(), ab = absorbers,
          bl = baseline, grid = fine) =
        build_forward_model(atm, ab, instrument, bl, space, units, m, x; wavenumber_fine_cm = grid)

    x_col = ComponentVector(H2O = 0.01 * n_air, CH4 = 1.9e-6 * n_air,
                            baseline = (window_1 = [1.0, 0.02, -0.01],))

    @testset "Jacobian against finite differences" begin
        x = ComponentVector(H2O = 0.01 * n_air, CH4 = 1.9e-6 * n_air, pressure_hPa = 850.0,
                            temperature_K = 290.0, instrument = (stretch = 1.0,),
                            baseline = (window_1 = [1.0, 0.02, -0.01],))
        model = build(x; instrument = DCS(calibration = LinearStretch()))
        K = ForwardDiff.jacobian(model, x)
        K_fd = finite_difference_jacobian(v -> model(ComponentVector(v, getaxes(x))), collect(x);
                                          rel_step = 1e-7)
        for j in axes(K, 2)
            scale = maximum(abs, K_fd[:, j])
            @test maximum(abs, K[:, j] - K_fd[:, j]) / scale < 1e-4
        end
    end

    @testset "baseline only without gas; absorption lowers the signal" begin
        model = build(x_col)
        x0 = copy(x_col); x0.H2O = 0.0; x0.CH4 = 0.0
        P = legendre_matrix(length(nu_obs), 3)
        @test isapprox(model(x0), P * x_col.baseline.window_1; rtol = 1e-12)
        @test all(model(x_col) .<= model(x0) .+ 1e-12)
    end

    @testset "column and VMR units give the same spectrum" begin
        x_vmr = column_to_vmr(x_col, homogeneous_air_column(850.0, 290.0, 195017.0);
                              gases = (:H2O, :CH4))
        @test isapprox(build(x_col)(x_col), build(x_vmr; units = VMR())(x_vmr); rtol = 1e-12)
    end

    @testset "Precomputed decided at build time" begin
        fixed = build(x_col)
        @test fixed.absorbers.H2O.model isa Precomputed
        @test fixed.absorbers.CH4.model isa Precomputed
        x_pt = ComponentVector(H2O = x_col.H2O, CH4 = x_col.CH4, pressure_hPa = 850.0,
                               temperature_K = 290.0, baseline = (window_1 = [1.0, 0.0, 0.0],))
        free = build(x_pt)
        @test !(free.absorbers.H2O.model isa Precomputed)
        @test !(free.absorbers.CH4.model isa Precomputed)
        x_pt.baseline.window_1 .= x_col.baseline.window_1
        @test isapprox(free(x_pt), fixed(x_col); rtol = 1e-12)
    end

    @testset "multi-window model with no absorbers returns the baselines" begin
        m1 = select_window(m, 6078.0, 6079.0)
        m2 = select_window(m, 6080.0, 6081.5)
        x = ComponentVector(baseline = (window_1 = [1.0, 0.1], window_2 = [2.0, -0.1, 0.05]))
        w1 = build_forward_model(atm, (;), DCS(), LegendreBaseline(2; name = :window_1),
                                 LinearSpace(), ColumnDensity(), m1, x; wavenumber_fine_cm = fine)
        w2 = build_forward_model(atm, (;), DCS(), LegendreBaseline(3; name = :window_2),
                                 LinearSpace(), ColumnDensity(), m2, x; wavenumber_fine_cm = fine)
        model = MultiWindowModel((w1, w2))
        expected = vcat(legendre_matrix(length(m1.y), 2) * [1.0, 0.1],
                        legendre_matrix(length(m2.y), 3) * [2.0, -0.1, 0.05])
        @test isapprox(model(x), expected; rtol = 1e-12)
        @test n_observations(model) == length(m1.y) + length(m2.y)
    end

    @testset "builder errors name the problem" begin
        missing_gas = ComponentVector(H2O = 1.0, baseline = (window_1 = [1.0, 0, 0],))
        @test_throws r"absorber CH4" build(missing_gas)
        extra = ComponentVector(H2O = 1.0, CH4 = 1.0, CO2 = 1.0, baseline = (window_1 = [1.0, 0, 0],))
        @test_throws r"CO2" build(extra)
        layered = ComponentVector(H2O = [1.0, 2.0], CH4 = 1.0, baseline = (window_1 = [1.0, 0, 0],))
        @test_throws r"x_prior.H2O" build(layered)
        @test_throws r"does not cover" build(x_col; grid = 6079.0:0.01:6083.0)
        short = ComponentVector(H2O = 1.0, CH4 = 1.0, baseline = (window_1 = [1.0, 0],))
        @test_throws r"baseline.window_1" build(short)
        @test_throws r"key does not match" build(x_col; ab = (H2O = absorbers.CH4, CH4 = absorbers.CH4))
    end

    @testset "type stability and allocations" begin
        model = build(x_col)
        @test @inferred(model(x_col)) isa Vector{Float64}
        dual_x = ComponentVector(ForwardDiff.Dual.(collect(x_col), 1.0), getaxes(x_col))
        @test eltype(@inferred(model(dual_x))) <: ForwardDiff.Dual
        model(x_col)
        @test (@allocated model(x_col)) < 2_000_000
        multi = MultiWindowModel((model, model))
        @test @inferred(multi(x_col)) isa Vector{Float64}
    end
end
