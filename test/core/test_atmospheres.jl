# Tests for src/core/atmospheres.jl: Beer's law, H2O rule, layered columns, Precomputed,
# parameter binding, type stability and ForwardDiff.

const ATM_CH4_LINE = AnalyticLorentzLine(6000.0, 1.0e-21, 1.0e-4)
const ATM_H2O_LINE = AnalyticLorentzLine(6000.3, 2.0e-22, 1.5e-4)

atm_dual_state(x) = ComponentVector(ForwardDiff.Dual{Nothing}.(collect(x), 1.0), getaxes(x))

@testset "HomogeneousPath construction and dry column" begin
    atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
    @test atm.pressure_hPa isa Fixed && atm.temperature_K isa Fixed
    @test n_layers(atm) == 1
    @test slant_factor(atm, 1) == 1.0
    @test dry_air_column(atm) == homogeneous_air_column(850.0, 290.0, 195017.0)
    x = ComponentVector(CH4 = 1.9e-6)
    @test dry_air_column(atm, x, 1, VMR()) == dry_air_column(atm)
    @test dry_air_column(atm, x, 1, ColumnDensity()) == dry_air_column(atm)
    @test layer_conditions(atm, x, 1, VMR()) == GasConditions(850.0, 290.0, 0.0)
    @test_throws ArgumentError HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 0.0)
    @test_throws ArgumentError HomogeneousPath(pressure_hPa = -1.0, temperature_K = 290.0, path_length_cm = 1.0)
    bound = bind_parameters(atm, ComponentVector(CH4 = 1.0, pressure_hPa = 800.0))
    @test bound.pressure_hPa isa FromState{:pressure_hPa}
    @test bound.temperature_K isa Fixed
    @test_throws ArgumentError dry_air_column(bound)
end

@testset "Beer's law on one Lorentz line" begin
    p, T = 850.0, 290.0
    atm = HomogeneousPath(pressure_hPa = p, temperature_K = T, path_length_cm = 1.0e5)
    gamma = ATM_CH4_LINE.hwhm_cm_per_hPa * p
    half_width = 50 * gamma
    nu = range(ATM_CH4_LINE.center_cm - half_width, ATM_CH4_LINE.center_cm + half_width; length = 1001)
    column = 1.0e20
    x = ComponentVector(CH4 = column)
    tau = optical_depth(atm, (CH4 = Absorber(:CH4, ATM_CH4_LINE),), ColumnDensity(), x, nu)
    sigma = cross_section(ATM_CH4_LINE, nu, GasConditions(p, T, 0.0))
    @test maximum(abs.(exp.(-tau) .- exp.(-column .* sigma))) == 0.0
    # trapezoid integral of absorbance; the Lorentz tails beyond the grid are known exactly
    h = step(nu)
    integral = h * (sum(tau) - (tau[1] + tau[end]) / 2)
    expected = column * ATM_CH4_LINE.strength * (2 / pi) * atan(half_width / gamma)
    @test isapprox(integral, expected; rtol = 1e-6)
    # bare models work as absorbers too
    @test optical_depth(atm, (CH4 = ATM_CH4_LINE,), ColumnDensity(), x, nu) == tau
    # no absorbers: zero optical depth
    @test all(iszero, optical_depth(atm, NamedTuple(), ColumnDensity(), x, nu))
end

@testset "H2O rule and column/VMR equivalence (homogeneous)" begin
    atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
    n_total = homogeneous_air_column(850.0, 290.0, 195017.0)
    nu = range(5999.0, 6001.0; length = 401)
    absorbers = (H2O = Absorber(:H2O, ATM_H2O_LINE), CH4 = Absorber(:CH4, ATM_CH4_LINE))
    x_vmr = ComponentVector(H2O = 0.012, CH4 = 1.9e-6)
    x_col = vmr_to_column(x_vmr, n_total; gases = (:H2O, :CH4))
    c_vmr = layer_conditions(atm, x_vmr, 1, VMR())
    c_col = layer_conditions(atm, x_col, 1, ColumnDensity())
    @test c_vmr.h2o_dry_mole_fraction == 0.012
    @test isapprox(c_col.h2o_dry_mole_fraction, 0.012; rtol = 1e-13)
    @test isapprox(c_col.h2o_dry_mole_fraction, x_col.H2O / (n_total - x_col.H2O); rtol = 1e-15)
    @test isapprox(dry_air_column(atm, x_col, 1, ColumnDensity()), n_total - x_col.H2O; rtol = 1e-14)
    @test isapprox(dry_air_column(atm, x_vmr, 1, VMR()), n_total / 1.012; rtol = 1e-15)
    @test isapprox(gas_column(VMR(), atm, x_vmr, :CH4, 1), x_col.CH4; rtol = 1e-13)
    @test gas_column(ColumnDensity(), atm, x_col, :CH4, 1) == x_col.CH4
    tau_vmr = optical_depth(atm, absorbers, VMR(), x_vmr, nu)
    tau_col = optical_depth(atm, absorbers, ColumnDensity(), x_col, nu)
    @test isapprox(tau_vmr, tau_col; rtol = 1e-12)
end

@testset "one-layer LayeredColumn equals HomogeneousPath" begin
    p, T, sza = 850.0, 290.0, 30.0
    hom = HomogeneousPath(pressure_hPa = p, temperature_K = T, path_length_cm = 195017.0)
    n_total = homogeneous_air_column(p, T, 195017.0)
    air_mass = 1 / cosd(sza)
    nu = range(5999.0, 6001.0; length = 401)
    absorbers = (H2O = Absorber(:H2O, ATM_H2O_LINE), CH4 = Absorber(:CH4, ATM_CH4_LINE))
    # VMR with H2O: same slant dry column N_total / (1 + q)
    q = 0.01
    lay = LayeredColumn(pressure_hPa = [p], temperature_K = [T], solar_zenith_deg = sza,
                        dry_air_column_cm2 = [n_total / (1 + q) / air_mass])
    @test n_layers(lay) == 1 && slant_factor(lay, 1) == air_mass
    x_hom = ComponentVector(H2O = q, CH4 = 1.9e-6)
    x_lay = ComponentVector(H2O = [q], CH4 = [1.9e-6])
    @test isapprox(optical_depth(lay, absorbers, VMR(), x_lay, nu),
                   optical_depth(hom, absorbers, VMR(), x_hom, nu); rtol = 1e-12)
    # column units: the layered state is vertical, the slant column is air_mass times it
    x_lay_col = ComponentVector(H2O = [1e22], CH4 = [4e19])
    x_hom_col = ComponentVector(H2O = air_mass * 1e22, CH4 = air_mass * 4e19)
    @test isapprox(optical_depth(lay, absorbers, ColumnDensity(), x_lay_col, nu),
                   optical_depth(hom, absorbers, ColumnDensity(), x_hom_col, nu); rtol = 1e-12)
    @test_throws ArgumentError LayeredColumn(pressure_hPa = [p], temperature_K = [T])
    @test_throws DimensionMismatch LayeredColumn(pressure_hPa = [p, 500.0], temperature_K = [T])
    @test_throws ArgumentError LayeredColumn(pressure_hPa = [p, 500.0], temperature_K = [T, T],
                                             solar_zenith_deg = 90.0)
end

layered_test_atmosphere() = LayeredColumn(pressure_hPa = collect(1000.0:-100.0:100.0),
                                          temperature_K = collect(range(290.0, 220.0; length = 10)),
                                          solar_zenith_deg = 40.0)

@testset "LayeredColumn hydrostatic columns and H2O" begin
    atm = layered_test_atmosphere()
    p = collect(1000.0:-100.0:100.0)
    dp = layer_pressure_thickness_hPa(p)
    x = ComponentVector(H2O = fill(1e21, 10), CH4 = fill(1.9e-6, 10))
    for l in (1, 5, 10)
        vertical = hydrostatic_dry_air_column(dp[l])
        @test dry_air_column(atm, x, l, VMR()) == atm.air_mass * vertical
        @test dry_air_column(atm, x, l, ColumnDensity()) == atm.air_mass * vertical
        @test layer_conditions(atm, x, l, ColumnDensity()).h2o_dry_mole_fraction == 1e21 / vertical
        @test layer_conditions(atm, x, l, ColumnDensity()).pressure_hPa == p[l]
        @test gas_column(ColumnDensity(), atm, x, :H2O, l) == atm.air_mass * 1e21
    end
    @test layer_conditions(atm, ComponentVector(CH4 = zeros(10)), 3, VMR()).h2o_dry_mole_fraction == 0.0
end

@testset "layered Precomputed equals direct per-layer loop" begin
    atm = layered_test_atmosphere()
    nu = range(5999.0, 6001.0; length = 801)
    models = (H2O = ATM_H2O_LINE, CH4 = ATM_CH4_LINE)
    for (units, x) in ((VMR(), ComponentVector(H2O = fill(0.005, 10), CH4 = collect(range(1.9e-6, 1.7e-6; length = 10)))),
                       (ColumnDensity(), ComponentVector(H2O = fill(1e22, 10), CH4 = fill(4e18, 10))))
        conditions = [layer_conditions(atm, x, l, units) for l in 1:n_layers(atm)]
        direct = zeros(length(nu))
        for l in 1:n_layers(atm), gas in (:H2O, :CH4)
            direct .+= gas_column(units, atm, x, gas, l) .*
                        cross_section(getproperty(models, gas), nu, conditions[l])
        end
        pre = map(m -> Absorber(:gas, precompute(m, nu, conditions)), models)
        @test pre.CH4.model isa Precomputed
        tau_pre = optical_depth(atm, pre, units, x, nu)
        tau_raw = optical_depth(atm, map(m -> Absorber(:gas, m), models), units, x, nu)
        @test isapprox(tau_pre, direct; rtol = 1e-12)
        @test isapprox(tau_raw, direct; rtol = 1e-12)
    end
end

@testset "parameter binding for LayeredColumn" begin
    atm = layered_test_atmosphere()
    p = collect(1000.0:-100.0:100.0)
    x = ComponentVector(CH4 = fill(1.9e-6, 10), pressure_hPa = p)
    bound = bind_parameters(atm, x)
    @test bound.pressure_hPa isa FromState{:pressure_hPa}
    @test bound.temperature_K isa Fixed
    @test bound.vertical_dry_air_column_cm2 isa SpectralFits.CoreLayer.HydrostaticFromState
    nu = range(5999.0, 6001.0; length = 101)
    ab = (CH4 = Absorber(:CH4, ATM_CH4_LINE),)
    @test isapprox(optical_depth(bound, ab, VMR(), x, nu), optical_depth(atm, ab, VMR(), x, nu); rtol = 1e-14)
    x2 = copy(x); x2.pressure_hPa .*= 0.9
    @test isapprox(dry_air_column(bound, x2, 4, VMR()), 0.9 * dry_air_column(atm, x, 4, VMR()); rtol = 1e-12)
    # explicit columns stay fixed when pressure is retrieved
    explicit = LayeredColumn(pressure_hPa = p, temperature_K = fill(250.0, 10),
                             dry_air_column_cm2 = fill(1e24, 10))
    @test bind_parameters(explicit, x).vertical_dry_air_column_cm2 == fill(1e24, 10)
end

@testset "type stability of optical_depth" begin
    nu = range(5999.0, 6001.0; length = 101)
    hom = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
    x = ComponentVector(H2O = 0.01, CH4 = 1.9e-6, pressure_hPa = 850.0, temperature_K = 290.0)
    hb = bind_parameters(hom, x)
    raw = (H2O = Absorber(:H2O, ATM_H2O_LINE), CH4 = Absorber(:CH4, ATM_CH4_LINE))
    conds = [layer_conditions(hom, x, 1, VMR())]
    pre = (H2O = Absorber(:H2O, precompute(ATM_H2O_LINE, nu, conds)),
           CH4 = Absorber(:CH4, precompute(ATM_CH4_LINE, nu, conds)))
    for units in (VMR(), ColumnDensity()), xx in (x, atm_dual_state(x))
        @test (@inferred optical_depth(hb, raw, units, xx, nu)) isa Vector{eltype(xx)}
        @test (@inferred optical_depth(hom, pre, units, xx, nu)) isa Vector{eltype(xx)}
    end
    lay = layered_test_atmosphere()
    xl = ComponentVector(H2O = fill(0.005, 10), CH4 = fill(1.9e-6, 10),
                         pressure_hPa = collect(1000.0:-100.0:100.0),
                         temperature_K = fill(250.0, 10))
    lb = bind_parameters(lay, xl)
    for xx in (xl, atm_dual_state(xl))
        @test (@inferred optical_depth(lb, raw, VMR(), xx, nu)) isa Vector{eltype(xx)}
        @test (@inferred optical_depth(lay, raw, ColumnDensity(), xx, nu)) isa Vector{eltype(xx)}
    end
end

@testset "ForwardDiff Jacobian of optical_depth vs finite differences" begin
    nu = range(5999.0, 6001.0; length = 201)
    ab = (H2O = Absorber(:H2O, ATM_H2O_LINE), CH4 = Absorber(:CH4, ATM_CH4_LINE))
    hom = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
    n_total = homogeneous_air_column(850.0, 290.0, 1e5)
    states = ((VMR(), ComponentVector(H2O = 0.01, CH4 = 1.9e-6, pressure_hPa = 850.0, temperature_K = 290.0)),
              (ColumnDensity(), ComponentVector(H2O = 0.01 * n_total, CH4 = 1.9e-6 * n_total,
                                                pressure_hPa = 850.0, temperature_K = 290.0)))
    for (units, x) in states
        atm = bind_parameters(hom, x)
        ax = getaxes(x)
        f = v -> optical_depth(atm, ab, units, ComponentVector(v, ax), nu)
        J_ad = ForwardDiff.jacobian(f, collect(x))
        J_fd = finite_difference_jacobian(f, collect(x); rel_step = 1e-6)
        for j in axes(J_ad, 2)
            scale = maximum(abs, J_fd[:, j])
            if scale == 0
                @test maximum(abs, J_ad[:, j]) == 0
            else
                @test maximum(abs, J_ad[:, j] .- J_fd[:, j]) <= 1e-5 * scale
            end
        end
        @test maximum(abs, J_ad[:, 3]) > 0   # pressure broadens the lines
        # the ComponentVector itself can be differentiated directly
        @test ForwardDiff.jacobian(v -> optical_depth(atm, ab, units, v, nu), x) ≈ J_ad
    end
    # VMR: temperature changes the dry-air column
    x = states[1][2]
    @test maximum(abs, ForwardDiff.jacobian(
        v -> optical_depth(bind_parameters(hom, x), ab, VMR(), ComponentVector(v, getaxes(x)), nu),
        collect(x))[:, 4]) > 0
    # layered column with a retrieved pressure profile
    lay = layered_test_atmosphere()
    xl = ComponentVector(CH4 = fill(1.9e-6, 10), pressure_hPa = collect(1000.0:-100.0:100.0))
    lb = bind_parameters(lay, xl)
    fl = v -> optical_depth(lb, (CH4 = Absorber(:CH4, ATM_CH4_LINE),), VMR(), ComponentVector(v, getaxes(xl)), nu)
    J_ad = ForwardDiff.jacobian(fl, collect(xl))
    J_fd = finite_difference_jacobian(fl, collect(xl); rel_step = 1e-6)
    @test maximum(abs, J_ad .- J_fd) <= 1e-5 * maximum(abs, J_fd)
end
