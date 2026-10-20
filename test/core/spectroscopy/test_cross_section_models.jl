# Phase 2 tests for HitranVoigt and LookupTable on the HITRAN 2020 subset (test/data).
import vSmartMOM.Absorption as Absorption

@testset "HitranVoigt" begin
    table = Absorption.read_hitran(FIXTURE_PAR; mol = 6, iso = 1, ν_min = 6076.0, ν_max = 6084.0)
    hitran_model = Absorption.make_hitran_model(table, Absorption.Voigt(); wing_cutoff = 40)
    ch4 = HitranVoigt(hitran_model; range_cm = (6076.0, 6084.0))
    grid = 6077.0:0.01:6083.0
    c = GasConditions(850.0, 290.0, 0.0)

    @testset "equals vSmartMOM to machine precision" begin
        expected = Absorption.absorption_cross_section(hitran_model, grid, 850.0, 290.0)
        sigma = cross_section(ch4, grid, c)
        @test sigma isa Vector{Float64}
        @test length(sigma) == length(grid)
        @test maximum(abs.(sigma .- expected)) <= 4 * eps(maximum(expected))
        @test maximum(sigma) > 1e-21          # CH4 lines of order 1e-20 cm2/molecule here
        # a vector grid gives the same answer as the range
        @test cross_section(ch4, collect(grid), c) == sigma
    end

    @testset "interface" begin
        @test depends_on(ch4) == (:pressure, :temperature)
        @test spectral_range_cm(ch4) == (6076.0, 6084.0)
        default_range = spectral_range_cm(HitranVoigt(hitran_model))
        @test default_range == (minimum(table.νᵢ), maximum(table.νᵢ))
        @test_throws ArgumentError HitranVoigt(hitran_model; range_cm = (6084.0, 6076.0))
        # H2O mole fraction has no effect on a bare HitranVoigt
        @test cross_section(ch4, grid, GasConditions(850.0, 290.0, 0.03)) == cross_section(ch4, grid, c)
    end

    @testset "ForwardDiff matches central differences in p and T" begin
        f(x) = cross_section(ch4, grid, GasConditions(x[1], x[2], 0.0))
        x0 = [850.0, 290.0]
        J_ad = ForwardDiff.jacobian(f, x0)
        J_fd = finite_difference_jacobian(f, x0; rel_step = 1e-6)
        @test size(J_ad) == (length(grid), 2)
        for j in 1:2   # column-norm relative error; elementwise fails where dsigma/dx ~ 0
            @test norm(J_ad[:, j] - J_fd[:, j]) / norm(J_fd[:, j]) < 1e-5
        end
        @test norm(J_ad[:, 1]) > 0 && norm(J_ad[:, 2]) > 0
    end

    @testset "type inference" begin
        @inferred cross_section(ch4, grid, c)
        D = ForwardDiff.Dual{Nothing,Float64,2}
        cd = GasConditions(ForwardDiff.Dual{Nothing}(850.0, 1.0, 0.0),
                           ForwardDiff.Dual{Nothing}(290.0, 0.0, 1.0),
                           ForwardDiff.Dual{Nothing}(0.0, 0.0, 0.0))
        sigma_dual = @inferred cross_section(ch4, grid, cd)
        @test sigma_dual isa Vector{D}
        # dual arithmetic may round differently from Float64 in the last bits
        sigma_f64 = cross_section(ch4, grid, c)
        @test maximum(abs.(ForwardDiff.value.(sigma_dual) .- sigma_f64)) <=
              1e-12 * maximum(sigma_f64)
        # integer conditions are promoted to Float64
        @test (@inferred cross_section(ch4, grid, GasConditions(850, 290, 0))) ==
              cross_section(ch4, grid, c)
    end
end

@testset "LookupTable" begin
    table = Absorption.read_hitran(FIXTURE_PAR; mol = 6, iso = 1, ν_min = 6076.0, ν_max = 6084.0)
    hitran_model = Absorption.make_hitran_model(table, Absorption.Voigt(); wing_cutoff = 40)
    ch4 = HitranVoigt(hitran_model; range_cm = (6076.0, 6084.0))
    nu_table = 6077.0:0.01:6083.0
    p_table = 800.0:25.0:900.0
    T_table = 280.0:5.0:300.0
    itp_model = Absorption.make_interpolation_model(table, Absorption.Voigt(), nu_table,
                                                    p_table, T_table; wing_cutoff = 40)
    lut = LookupTable(itp_model)
    grid = 6078.0:0.01:6082.0       # table wavenumber nodes inside the grid

    @testset "interface" begin
        @test depends_on(lut) == (:pressure, :temperature)
        @test spectral_range_cm(lut) == (6077.0, 6083.0)
    end

    @testset "agrees with HitranVoigt" begin
        # at table nodes the cubic B-spline interpolates the Voigt values
        for (p, T) in ((800.0, 280.0), (850.0, 290.0), (900.0, 300.0))
            c = GasConditions(p, T, 0.0)
            sigma_voigt = cross_section(ch4, grid, c)
            @test maximum(abs.(cross_section(lut, grid, c) .- sigma_voigt)) <=
                  1e-10 * maximum(sigma_voigt)
        end
        # between nodes: interpolation tolerance 5e-5 of the peak, set for this
        # 25 hPa by 5 K table (the observed error is about 1e-5)
        for (p, T) in ((840.0, 287.0), (812.5, 282.5), (887.0, 298.0))
            c = GasConditions(p, T, 0.0)
            sigma_voigt = cross_section(ch4, grid, c)
            @test maximum(abs.(cross_section(lut, grid, c) .- sigma_voigt)) <=
                  5e-5 * maximum(sigma_voigt)
        end
    end

    @testset "derivatives and inference" begin
        c = GasConditions(840.0, 287.0, 0.0)
        @inferred cross_section(lut, grid, c)
        D = ForwardDiff.Dual{Nothing,Float64,2}
        cd = GasConditions(ForwardDiff.Dual{Nothing}(840.0, 1.0, 0.0),
                           ForwardDiff.Dual{Nothing}(287.0, 0.0, 1.0),
                           ForwardDiff.Dual{Nothing}(0.0, 0.0, 0.0))
        @test (@inferred cross_section(lut, grid, cd)) isa Vector{D}
        f(x) = cross_section(lut, grid, GasConditions(x[1], x[2], 0.0))
        J_ad = ForwardDiff.jacobian(f, [840.0, 287.0])
        J_fd = finite_difference_jacobian(f, [840.0, 287.0]; rel_step = 1e-6)
        for j in 1:2
            @test norm(J_ad[:, j] - J_fd[:, j]) / norm(J_fd[:, j]) < 1e-5
        end
    end

    @testset "outside the table is an error" begin
        @test_throws DomainError cross_section(lut, grid, GasConditions(950.0, 290.0, 0.0))
        @test_throws DomainError cross_section(lut, grid, GasConditions(850.0, 310.0, 0.0))
        @test_throws DomainError cross_section(lut, 6076.0:0.01:6080.0, GasConditions(850.0, 290.0, 0.0))
    end
end
