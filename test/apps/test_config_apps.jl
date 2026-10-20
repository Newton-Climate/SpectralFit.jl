# Apps run from TOML configuration files (Apps.run_config): batch retrieval and profile OSSE.
@isdefined(write_batch_dcs_file) || include(joinpath(@__DIR__, "test_batch_retrieval.jl"))
import SpectralFits.Apps.ProfileOSSE as PO

@testset "batch retrieval from a configuration file, with a conditions table" begin
    mktempdir() do dir
        data = mkpath(joinpath(dir, "data"))
        write_batch_dcs_file(joinpath(data, "20160921a.h5"), 2.0e-6; seed = 1)
        # the averaged measurements start at 00:00, 00:02, 00:04 (first samples 00:00:00, 00:02:00, 00:04:00)
        write(joinpath(dir, "tower.csv"), "time,pressure_hPa\n" *
              join(["2016-09-21T00:0$(m):05,850.0" for m in (0, 2, 4)], "\n") * "\n")
        write(joinpath(dir, "run.toml"), """
            app = "batch_retrieval"
            experiment = "toml-test"
            data_directory = "data"
            output_root = "runs"
            averaging_minutes = 2
            parallel = "serial"
            [reader]
            path_length_cm = 195017.0
            [noise]
            model = "constant"
            variance = 1e-6
            [conditions]
            file = "tower.csv"
            max_offset_s = 30
            [absorbers.CH4]
            file = "$(FIXTURE_PAR)"
            molecule = 6
            range_cm = [6074.0, 6086.0]
            [retrieval]
            window_cm = [6076.5, 6077.5]
            baseline_coefficients = 3
            fine_grid_spacing_cm = 0.002
            [solver]
            max_iterations = 10
            [prior]
            reference_pressure_hPa = 850.0
            reference_temperature_K = 288.15
            baseline_sigma = 10.0
            [prior.gases]
            CH4 = { value = 1.9e-6, sigma = 1e-6 }
            """)
        config = only(SpectralFits.Apps.run_config(joinpath(dir, "run.toml"); dry_run = true))
        @test config isa BatchRetrievalConfig
        @test config.x_prior.CH4 ≈ 1.9e-6 * BATCH_N_AIR
        @test config.conditions isa ConditionsTable
        summary = only(SpectralFits.Apps.run_config(joinpath(dir, "run.toml")))
        @test summary isa BatchRetrievalSummary
        @test SF.find_run(joinpath(dir, "runs"), "toml-test") == summary.run_folder
        loaded = load_results(only(summary.result_files))
        @test length(loaded.results) == 3
        r = loaded.results[1]
        @test r.status == Converged
        @test isapprox(r.x.CH4 / BATCH_N_AIR, 2.0e-6; rtol = 0.02)
        @test r.coordinate ≈ filter(v -> 6076.5 <= v <= 6077.5, BATCH_NU) rtol = 1e-12
        @test r.metadata["conditions"]["pressure_hPa"] == 850.0
        @test r.metadata["applied_conditions"]["pressure_hPa"] == 850.0
    end
end

@testset "profile OSSE from a configuration file with variants" begin
    mktempdir() do dir
        write_batch_dcs_file(joinpath(dir, "template.h5"), 2.0e-6; seed = 3)
        write(joinpath(dir, "osse.toml"), """
            app = "profile_osse"
            experiment = "ch4"
            output_root = "runs"
            window_cm = [6076.5, 6077.5]
            fine_grid_spacing_cm = 0.002
            measurement_space = "log"
            baseline_coefficients = 3
            target = "CH4"
            [template]
            dcs_file = "template.h5"
            average_index = 1
            averaging_minutes = 2
            noise = { model = "constant", variance = 1e-6 }
            [noise]
            source = "snr"
            snr = 20000.0
            [atmosphere]
            n_layers = 5
            top_m = 8000.0
            pressure = { kind = "us_standard" }
            temperature = { kind = "lapse_rate", surface_K = 288.15, lapse_K_per_m = 0.0065 }
            [gases.H2O]
            truth = { kind = "exponential", surface = 0.006, scale_height_m = 2000.0 }
            prior = 0.004
            sigma = 0.01
            cross_section = { file = "$(FIXTURE_PAR)", molecule = 1, range_cm = [6076.5, 6077.5] }
            [gases.CH4]
            truth = { kind = "linear", bottom = 2000e-9, top = 1900e-9 }
            prior = 1950e-9
            sigma = 300e-9
            correlation_length = 2.0
            cross_section = { file = "$(FIXTURE_PAR)", molecule = 6, range_cm = [6076.5, 6077.5] }
            [prior]
            correlation_coordinate = "height_km"
            [solver]
            name = "levenberg_marquardt"
            initial_damping = 1.0
            [[variants]]
            name = "profile"
            [[variants]]
            name = "isotherm"
            atmosphere.mode = "isotherm"
            """)
        configs = SpectralFits.Apps.run_config(joinpath(dir, "osse.toml"); dry_run = true)
        @test [c.experiment for c in configs] == ["ch4_profile", "ch4_isotherm"]
        atm = PO.profile_atmosphere(configs[2])
        @test all(atm.temperature_K .== atm.temperature_K[1])
        @test atm.pressure_hPa == PO.profile_atmosphere(configs[1]).pressure_hPa
        summaries = SpectralFits.Apps.run_config(joinpath(dir, "osse.toml"))
        for s in summaries
            @test s isa PO.ProfileOSSESummary
            @test s.status == Converged
            @test abs(s.column_error_percent[:CH4]) < 3 * s.column_sigma_percent[:CH4] + 0.1
            @test 0 < s.dfs[:CH4] <= 5
            @test isfile(joinpath(s.run_folder, "configuration.toml")) && isfile(joinpath(s.run_folder, "summary.txt"))
            loaded = load_results(s.results_file)
            x_true = state_from_labels(loaded.attributes["x_true"], loaded.attributes["x_true_labels"])
            @test length(x_true.CH4) == 5
            @test occursin("gas,dfs,column_error_percent", report(s))
        end
    end
end
