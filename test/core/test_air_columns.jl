@testset "Loschmidt number" begin
    @test isapprox(number_density_per_cm3(1013.25, 273.15), 2.6867811e19; rtol = 1e-3)
    @test homogeneous_air_column(1013.25, 273.15, 1.0) == number_density_per_cm3(1013.25, 273.15)
end

@testset "hydrostatic layer column" begin
    delta_p = 12.5
    m_dry = MOLAR_MASS_DRY_AIR_KG_PER_MOL / AVOGADRO_PER_MOL
    @test hydrostatic_dry_air_column(delta_p) == delta_p * 100 / (m_dry * GRAVITY_M_PER_S2) / 1e4
end

@testset "layer pressure thickness" begin
    p = collect(1000.0:-50.0:500.0)
    dp = layer_pressure_thickness_hPa(p)
    @test length(dp) == length(p)
    @test all(isapprox.(dp, 50.0; rtol = 1e-14))
    @test_throws ArgumentError layer_pressure_thickness_hPa([1000.0])
end

@testset "column and VMR states round trip" begin
    n_total = homogeneous_air_column(850.0, 290.0, 195017.0)
    for with_h2o in (true, false)
        x_vmr = with_h2o ?
            ComponentVector(H2O = 0.012, CH4 = 1.9e-6, CO2 = 410e-6, pressure_hPa = 850.0) :
            ComponentVector(CH4 = 1.9e-6, CO2 = 410e-6, pressure_hPa = 850.0)
        gases = with_h2o ? (:H2O, :CH4, :CO2) : (:CH4, :CO2)
        x_col = vmr_to_column(x_vmr, n_total; gases)
        back = column_to_vmr(x_col, n_total; gases)
        @test all(isapprox.(back, x_vmr; rtol = 1e-14))
        @test back.pressure_hPa == 850.0
        # dry column consistency: N_total = N_dry + N_H2O
        n_h2o = with_h2o ? x_col.H2O : 0.0
        @test isapprox(x_col.CH4 / x_vmr.CH4 + n_h2o, n_total; rtol = 1e-14)
    end
end

@testset "prior precision from sigma" begin
    sigma = ComponentVector(CH4 = 2.0, baseline = (w = [1.0, 0.5],))
    P = prior_precision_from_sigma(sigma)
    @test P == Diagonal([0.25, 1.0, 4.0])
    @test_throws ArgumentError prior_precision_from_sigma(ComponentVector(CH4 = 0.0))
    @test_throws ArgumentError prior_precision_from_sigma(sigma, ComponentVector(CH4 = 1.0, H2O = 1.0, b = 1.0))
end
