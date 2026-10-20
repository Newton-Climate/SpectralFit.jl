# Phase 2 tests for H2OBroadened, Precomputed (built by precompute) and ABSCOTable.
import vSmartMOM.Absorption as Absorption

@testset "H2OBroadened" begin
    ch4 = SF.IOLayer.load_cross_section_model(HitranVoigt, FIXTURE_PAR; molecule = 6,
                                              isotope = 1, range_cm = (6076.0, 6084.0))
    lorentz = AnalyticLorentzLine(6080.0, 1e-20, 1e-4)
    grid = 6077.0:0.01:6083.0
    p, T, q, coef = 850.0, 290.0, 0.02, 0.34

    for inner in (ch4, lorentz)
        @test cross_section(H2OBroadened(inner, 0.0), grid, GasConditions(p, T, q)) ==
              cross_section(inner, grid, GasConditions(p, T, q))
        wet = H2OBroadened(inner, coef)
        @test cross_section(wet, grid, GasConditions(p, T, q)) ==
              cross_section(inner, grid, GasConditions(p * (1 + coef * q), T, q))
        @test spectral_range_cm(wet) == spectral_range_cm(inner)
    end

    @testset "depends_on" begin
        @test depends_on(H2OBroadened(ch4, coef)) == (:pressure, :temperature, :h2o)
        @test depends_on(H2OBroadened(lorentz, coef)) == (:pressure, :h2o)
        @test depends_on(H2OBroadened(H2OBroadened(lorentz, coef), coef)) == (:pressure, :h2o)
    end

    @testset "derivative in H2O" begin
        # d sigma / d q = d sigma / d p_eff * p * c, checked on the analytic line
        wet = H2OBroadened(lorentz, coef)
        g(x) = cross_section(wet, grid, GasConditions(p, T, x[1]))
        J_ad = ForwardDiff.jacobian(g, [q])
        J_fd = finite_difference_jacobian(g, [q]; rel_step = 1e-6)
        @test norm(J_ad - J_fd) / norm(J_fd) < 1e-6
        h(x) = cross_section(lorentz, grid, GasConditions(x[1], T, q))
        J_p = ForwardDiff.jacobian(h, [p * (1 + coef * q)])
        @test isapprox(J_ad, J_p .* (p * coef); rtol = 1e-12)
    end

    @testset "type inference" begin
        wet = H2OBroadened(ch4, coef)
        @inferred cross_section(wet, grid, GasConditions(p, T, q))
        D = ForwardDiff.Dual{Nothing,Float64,3}
        cd = GasConditions(ForwardDiff.Dual{Nothing}(p, 1.0, 0.0, 0.0),
                           ForwardDiff.Dual{Nothing}(T, 0.0, 1.0, 0.0),
                           ForwardDiff.Dual{Nothing}(q, 0.0, 0.0, 1.0))
        @test (@inferred cross_section(wet, grid, cd)) isa Vector{D}
    end
end

@testset "Precomputed equals inner model" begin
    ch4 = SF.IOLayer.load_cross_section_model(HitranVoigt, FIXTURE_PAR; molecule = 6,
                                              isotope = 1, range_cm = (6076.0, 6084.0))
    grid = 6077.0:0.01:6083.0
    layers = [GasConditions(850.0, 290.0, 0.01), GasConditions(700.0, 270.0, 0.005)]
    for inner in (ch4, H2OBroadened(ch4, 0.34))
        pre = precompute(inner, grid, layers)
        @test pre isa Precomputed
        @test depends_on(pre) == ()
        for (layer, c) in enumerate(layers)
            @test layer_cross_section(pre, grid, c, layer) == cross_section(inner, grid, c)
        end
        single = precompute(inner, grid, layers[1:1])
        @test cross_section(single, grid, layers[2]) == cross_section(inner, grid, layers[1])
    end
end

@testset "ABSCOTable" begin
    # synthetic table: sigma = f(nu, q_wet, T, p), linear in each axis, with cross terms
    f(nu, q, T, p) = 1e-22 * (1 + 0.5 * (nu - 6200) + 3q + 0.01 * (T - 250) +
                              2e-3 * p + 1e-5 * (T - 250) * p + 0.2 * q * (nu - 6200))
    nu = collect(6200.0:0.5:6205.0)
    q_axis = [0.0, 0.02, 0.05]
    p_axis = [100.0, 400.0, 700.0, 1000.0]
    # per-level temperature grids, as in real ABSCO files: warmer at higher pressure
    T_axis = hcat([collect(180.0:20.0:300.0) .+ 0.02 * pk for pk in p_axis]...)
    sigma = [f(nu[i], q_axis[j], T_axis[k, l], p_axis[l])
             for i in eachindex(nu), j in eachindex(q_axis), k in axes(T_axis, 1),
                 l in eachindex(p_axis)]
    table = ABSCOTable(sigma, (wavenumber_cm = nu, h2o_mole_fraction = q_axis,
                               temperature_K = T_axis, pressure_hPa = p_axis))
    grid = 6200.1:0.13:6204.9

    @testset "interface" begin
        @test depends_on(table) == (:pressure, :temperature, :h2o)
        @test spectral_range_cm(table) == (6200.0, 6205.0)
    end

    @testset "reproduces a function linear in each axis" begin
        for (p, T, q_dry) in ((850.0, 290.0, 0.01), (100.0, 190.0, 0.0), (433.3, 251.7, 0.04),
                              (1000.0, 312.0, 0.04 / 0.96))
            q_wet = q_dry / (1 + q_dry)
            expected = f.(grid, q_wet, T, p)
            got = cross_section(table, grid, GasConditions(p, T, q_dry))
            @test maximum(abs.(got .- expected)) <= 1e-12 * maximum(abs.(expected))
        end
        # exactly on the wavenumber nodes
        c = GasConditions(550.0, 260.0, 0.03)
        @test isapprox(cross_section(table, nu, c), f.(nu, 0.03 / 1.03, 260.0, 550.0); rtol = 1e-12)
    end

    @testset "derivatives are exact for the linear table" begin
        g(x) = cross_section(table, grid, GasConditions(x[1], x[2], x[3]))
        x0 = [633.0, 270.0, 0.015]
        J_ad = ForwardDiff.jacobian(g, x0)
        J_fd = finite_difference_jacobian(g, x0; rel_step = 1e-6)
        for j in 1:3
            @test norm(J_ad[:, j] - J_fd[:, j]) / norm(J_fd[:, j]) < 1e-6
        end
        # analytic d sigma / d p at fixed q_wet and T
        @test isapprox(J_ad[:, 1], fill(1e-22 * (2e-3 + 1e-5 * (270.0 - 250)), length(grid));
                       rtol = 1e-9)
    end

    @testset "shared temperature vector and single H2O value" begin
        T_vec = collect(200.0:25.0:300.0)
        s2 = [f(nu[i], 0.0, T_vec[k], p_axis[l])
              for i in eachindex(nu), j in 1:1, k in eachindex(T_vec), l in eachindex(p_axis)]
        t2 = ABSCOTable(s2, (wavenumber_cm = nu, h2o_mole_fraction = [0.0],
                             temperature_K = T_vec, pressure_hPa = p_axis))
        @test size(t2.temperature_K) == (length(T_vec), length(p_axis))
        # with one H2O value the table is held constant in H2O
        got = cross_section(t2, grid, GasConditions(500.0, 260.0, 0.02))
        @test isapprox(got, f.(grid, 0.0, 260.0, 500.0); rtol = 1e-12)
    end

    @testset "errors" begin
        @test_throws DomainError cross_section(table, grid, GasConditions(50.0, 250.0, 0.0))
        @test_throws DomainError cross_section(table, grid, GasConditions(500.0, 400.0, 0.0))
        @test_throws DomainError cross_section(table, grid, GasConditions(500.0, 250.0, 0.2))
        @test_throws DomainError cross_section(table, 6199.0:0.1:6201.0, GasConditions(500.0, 250.0, 0.0))
        bad_axes = (wavenumber_cm = nu, h2o_mole_fraction = q_axis, temperature_K = T_axis,
                    pressure_hPa = reverse(p_axis))
        @test_throws ArgumentError ABSCOTable(sigma, bad_axes)
        @test_throws DimensionMismatch ABSCOTable(sigma[1:end-1, :, :, :],
            (wavenumber_cm = nu, h2o_mole_fraction = q_axis, temperature_K = T_axis,
             pressure_hPa = p_axis))
    end

    @testset "type inference" begin
        c = GasConditions(850.0, 290.0, 0.01)
        @inferred cross_section(table, grid, c)
        D = ForwardDiff.Dual{Nothing,Float64,3}
        cd = GasConditions(ForwardDiff.Dual{Nothing}(850.0, 1.0, 0.0, 0.0), ForwardDiff.Dual{Nothing}(290.0, 0.0, 1.0, 0.0),
                        ForwardDiff.Dual{Nothing}(0.01, 0.0, 0.0, 1.0))
        @test (@inferred cross_section(table, grid, cd)) isa Vector{D}
    end
end
