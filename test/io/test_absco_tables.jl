# Phase 2 tests for read_absco_table on small HDF5 fixtures written here, in the ABSCO layout.
import HDF5

"""Write an ABSCO-like HDF5 file; arrays are in the Julia order HDF5.jl reads real files in."""
function write_absco_fixture(path; nu, h2o, T, p_Pa, sigma, gas = 2, extra_gas = false)
    HDF5.h5open(path, "w") do file
        file["Wavenumber"] = nu
        h2o === nothing || (file["Broadener_01_VMR"] = h2o)
        file["Temperature"] = T
        file["Pressure"] = p_Pa
        file[string("Gas_", lpad(gas, 2, '0'), "_Absorption")] = sigma
        extra_gas && (file["Gas_07_Absorption"] = sigma)
    end
    return path
end

@testset "read_absco_table" begin
    read_absco = SF.IOLayer.read_absco_table
    nu = collect(6200.0:0.25:6210.0)
    h2o = [0.0, 0.03]
    p_Pa = [10000.0, 50000.0, 100000.0]
    T = hcat([collect(190.0:30.0:310.0) .+ 0.01 * (pk / 100) for pk in p_Pa]...)  # (n_T, n_p)
    f(nu, q, T, p_hPa) = 1e-23 * (2 + 0.1 * (nu - 6200) + 4q + 0.01 * T + 1e-3 * p_hPa)
    sigma = [f(nu[i], h2o[j], T[k, l], p_Pa[l] / 100)
             for i in eachindex(nu), j in eachindex(h2o), k in axes(T, 1), l in eachindex(p_Pa)]

    mktempdir() do dir
        path = write_absco_fixture(joinpath(dir, "co2.h5"); nu, h2o, T, p_Pa, sigma)

        @testset "h2o_indices keeps a slice of the broadener axis" begin
            dry = SF.IOLayer.read_absco_table(path; h2o_indices = [1])
            @test dry.h2o_mole_fraction == [0.0]
            @test dry.sigma == sigma[:, 1:1, :, :]
            # an axis of length one is held constant: any H2O gives the dry values
            c = GasConditions(700.0, 260.0, 0.02)
            @test cross_section(dry, 6201.0:0.1:6209.0, c) ≈ f.(6201.0:0.1:6209.0, 0.0, 260.0, 700.0) rtol = 1e-10
            @test_throws r"h2o_indices" SF.IOLayer.read_absco_table(path; h2o_indices = [3])
        end

        @testset "returns the table written" begin
            table = read_absco(path)
            @test table isa ABSCOTable
            @test table.wavenumber_cm == nu
            @test table.h2o_mole_fraction == h2o
            @test table.temperature_K == T
            @test table.pressure_hPa == p_Pa ./ 100
            @test table.sigma == sigma
            @test spectral_range_cm(table) == (6200.0, 6210.0)
            c = GasConditions(700.0, 260.0, 0.01)
            expected = f.(6201.0:0.1:6209.0, 0.01, 260.0, 700.0)        # dry q, no conversion
            @test isapprox(cross_section(table, 6201.0:0.1:6209.0, c), expected; rtol = 1e-12)
            @test read_absco(path; gas = 2).sigma == sigma
        end

        @testset "wavenumber subset" begin
            sub = read_absco(path; range_cm = (6203.1, 6205.4))
            @test sub.wavenumber_cm == nu[(nu .>= 6203.0) .& (nu .<= 6205.5)]
            @test spectral_range_cm(sub)[1] <= 6203.1 && spectral_range_cm(sub)[2] >= 6205.4
            idx = findfirst(==(6203.0), nu):findfirst(==(6205.5), nu)
            @test sub.sigma == sigma[idx, :, :, :]
            # exactly on nodes
            sub2 = read_absco(path; range_cm = (6203.0, 6205.0))
            @test sub2.wavenumber_cm == collect(6203.0:0.25:6205.0)
            @test_throws ArgumentError read_absco(path; range_cm = (6190.0, 6205.0))
        end

        @testset "no broadener axis, shared temperatures" begin
            T_vec = collect(200.0:20.0:320.0)
            s3 = [f(nu[i], 0.0, T_vec[k], p_Pa[l] / 100)
                  for i in eachindex(nu), k in eachindex(T_vec), l in eachindex(p_Pa)]
            path3 = write_absco_fixture(joinpath(dir, "ch4.h5"); nu, h2o = nothing, T = T_vec,
                                        p_Pa, sigma = s3, gas = 6)
            t3 = read_absco(path3)
            @test t3.h2o_mole_fraction == [0.0]
            @test size(t3.sigma) == (length(nu), 1, length(T_vec), length(p_Pa))
            @test t3.temperature_K == repeat(T_vec, 1, length(p_Pa))
            @test isapprox(cross_section(t3, 6201.0:0.5:6209.0, GasConditions(400.0, 250.0, 0.0)),
                           f.(6201.0:0.5:6209.0, 0.0, 250.0, 400.0); rtol = 1e-12)
        end

        @testset "errors" begin
            two = write_absco_fixture(joinpath(dir, "two.h5"); nu, h2o, T, p_Pa, sigma,
                                      extra_gas = true)
            @test_throws ArgumentError read_absco(two)          # ambiguous gas
            @test read_absco(two; gas = 7).sigma == sigma
            @test_throws ArgumentError read_absco(path; gas = 6)
            @test_throws ArgumentError read_absco(joinpath(dir, "missing.h5"))
            HDF5.h5open(joinpath(dir, "bad.h5"), "w") do file
                file["Wavenumber"] = nu
            end
            @test_throws ArgumentError read_absco(joinpath(dir, "bad.h5"))
        end
    end
end
