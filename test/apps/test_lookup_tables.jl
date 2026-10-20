# Lookup-table app and the core tabulation it uses.
import SpectralFits.Apps.LookupTables as LT
import TOML

@testset "tabulate and spline a cross-section model" begin
    m = fixture_voigt(:CH4, (6076.0, 6080.0))
    nu, p, T = 6077.0:0.01:6079.0, 700.0:50.0:900.0, 260.0:10.0:300.0
    cs = tabulate_cross_sections(m, nu, p, T)
    @test size(cs) == (length(nu), length(p), length(T))
    @test cs[:, 2, 3] ≈ cross_section(m, nu, GasConditions(750.0, 280.0, 0.0)) rtol = 1e-12
    t = lookup_table_from_grid(cs, nu, p, T; molecule = 6, isotope = 1)
    @test cross_section(t, nu, GasConditions(800.0, 270.0, 0.0)) ≈ cs[:, 3, 2] rtol = 1e-10   # exact at nodes
    @test_throws r"outside the model range" tabulate_cross_sections(m, 6070.0:0.01:6079.0, p, T)
    @test_throws DimensionMismatch lookup_table_from_grid(cs[:, 1:2, :], nu, p, T)
end

@testset "lookup_table app from a configuration file" begin
    mktempdir() do dir
        write(joinpath(dir, "t.toml"), """
            app = "lookup_table"
            experiment = "ch4_test"
            output_file = "out/ch4_test.jld2"
            wavenumber_cm = { start = 6077.0, stop = 6079.0, step = 0.01 }
            pressure_hPa = { start = 700.0, stop = 900.0, step = 50.0 }
            temperature_K = { start = 260.0, stop = 300.0, step = 10.0 }
            coefficient_type = "Float32"
            [source]
            file = "$(FIXTURE_PAR)"
            molecule = 6
            line_margin_cm = 1.0
            """)
        s = only(SpectralFits.Apps.run_config(joinpath(dir, "t.toml")))
        @test s isa LT.LookupTableSummary
        @test isfile(joinpath(dir, "out", "ch4_test.jld2")) && isfile(joinpath(dir, "out", "ch4_test.toml"))
        @test 0 < s.max_relative_error < 0.05
        t = load_cross_section_model(LookupTable, joinpath(dir, "out", "ch4_test.jld2"))
        @test spectral_range_cm(t) == (6077.0, 6079.0)
        @test eltype(t.model.itp.coefs) == Float32
        @test TOML.parsefile(joinpath(dir, "out", "ch4_test.toml"))["experiment"] == "ch4_test"
        # refuses to overwrite unless asked
        @test_throws r"overwrite" SpectralFits.Apps.run_config(joinpath(dir, "t.toml"); dry_run = true)
        # relabelled line list: the fixture's CH4 lines relabelled as molecule 2 are read as CO2
        sub = subset_line_list(FIXTURE_PAR, joinpath(dir, "sub.par"); range_cm = (6076.0, 6080.0), relabel = " 61" => " 21")
        @test all(l -> l[1:3] == " 21", readlines(sub))
        @test_throws r"no lines" subset_line_list(FIXTURE_PAR, joinpath(dir, "none.par"); range_cm = (1.0, 2.0))
    end
end

@isdefined(batch_test_config) || include(joinpath(@__DIR__, "test_batch_retrieval.jl"))
@testset "batch configuration checks table coverage of the fine grid" begin
    narrow = (CH4 = Absorber(:CH4, fixture_voigt(:CH4, (6076.0, 6078.0))),)
    window = RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [3])
    margin = RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [3],
                             fine_grid_margin_cm = 0.04)
    mktempdir() do dir
        @test_throws r"fine_grid_margin_cm" batch_test_config(data_directory = dir, absorbers = narrow, retrieval = window)
        @test batch_test_config(data_directory = dir, absorbers = narrow, retrieval = margin) isa BatchRetrievalConfig
    end
end
