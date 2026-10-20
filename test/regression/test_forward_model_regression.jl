# Phase 4 regression: spectra match the v0.1 reference (test/regression/README.md).
using JLD2

let ref = JLD2.load(joinpath(@__DIR__, "reference_v0_1.jld2"))
    for (window, gases) in (("ch4", (:H2O, :CH4)), ("co2", (:H2O, :CO2)))
        g = window * "_fm"
        fine = ref["$g/fine_grid_start"]:ref["$g/fine_grid_step"]:ref["$g/fine_grid_stop"]
        range_cm = (first(fine), last(fine))
        absorbers = NamedTuple{gases}(map(gas -> Absorber(gas, fixture_voigt(gas, range_cm)), gases))
        atm = HomogeneousPath(pressure_hPa = ref["$g/pressure_hPa"], temperature_K = ref["$g/temperature_K"],
                              path_length_cm = ref["$g/pathlength_cm"])
        nu = ref["$g/meas_grid"]
        m = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1.0)

        @testset "$window cross sections" begin
            for gas in gases
                sigma = cross_section(absorbers[gas].model, fine, GasConditions(atm.pressure_hPa.value, atm.temperature_K.value, 0.0))
                @test isapprox(sigma, ref["$g/xsec_$(gas)_cm2_per_molec"]; rtol = 1e-10)
            end
        end

        for (variant, space) in (("intensity", LinearSpace()), ("log", LogSpace()))
            key = "$(g)_$variant"
            haskey(ref, "$key/spectrum") || continue
            shape = ref["$key/shape_parameters"]
            gas_values = NamedTuple{gases}(map(gas -> ref["$key/state_$(gas)_molec_per_cm2"], gases))
            x = ComponentVector(merge(gas_values, (baseline = (window_1 = shape,),)))
            model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(length(shape)), space,
                                        ColumnDensity(), m, x; wavenumber_fine_cm = fine)
            @testset "$window $variant spectrum" begin
                @test isapprox(model(x), ref["$key/spectrum"]; rtol = 1e-10)
            end
        end
    end
end
