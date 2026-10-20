# Phase 9: batch retrieval app on synthetic DCS HDF5 files.
using HDF5, JLD2, Distributed
import TOML
using Logging
import SpectralFits.Apps.BatchRetrieval as BR
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, BatchRetrievalSummary

const BATCH_NU = collect(6076.3:0.0067:6077.7)
const BATCH_CONDITIONS = (pressure_hPa = 850.0, temperature_K = 288.15, path_length_cm = 195017.0)
const BATCH_N_AIR = homogeneous_air_column(850.0, 288.15, 195017.0)
const BATCH_ABSORBERS = (CH4 = Absorber(:CH4, fixture_voigt(:CH4, (6074.0, 6086.0))),)
const BATCH_T0 = DateTime(2016, 9, 21, 0, 0, 0)

"""Noise-free transmitted intensity on BATCH_NU for a CH4 dry mole fraction (mol/mol)."""
function batch_true_intensity(ch4_vmr)
    m = Measurement(y = ones(length(BATCH_NU)), coordinate = BATCH_NU, noise_variance = 1e-6,
                    conditions = BATCH_CONDITIONS)
    config = RetrievalConfig(windows_cm = [(6076.2, 6077.8)], baseline_coefficients = [2],
                             fine_grid_spacing_cm = 0.002)
    x = ComponentVector(CH4 = ch4_vmr * BATCH_N_AIR, baseline = (window_1 = [1.0, 0.02],))
    model = SpectralFits.CoreLayer.build_forward_model(config, BATCH_ABSORBERS, m, x)
    return model(x)
end

"""
Write a DCS file (Pres_CS / Temp_CS / TotalTime layout) with 9 records 40 s apart from
BATCH_T0, so 2-minute averaging gives 3 measurements. `nan_record` gives one record a NaN pressure:
the reader accepts it (intensity and noise are valid), so the averaged measurement reaches the
retrieval and fails there, which exercises the app-level failure path.
"""
function write_batch_dcs_file(path, ch4_vmr; seed, noise_sigma = 1e-3, nan_record = nothing)
    rng = MersenneTwister(seed)
    truth = batch_true_intensity(ch4_vmr)
    n_records = 9
    intensity = reduce(vcat, ((truth .+ noise_sigma .* randn(rng, length(truth)))'
                              for _ in 1:n_records))
    h5open(path, "w") do f
        pressure = fill(850.0, n_records)
        nan_record === nothing || (pressure[nan_record] = NaN)
        f["Pres_CS"] = pressure
        f["Temp_CS"] = fill(15.0, n_records)
        f["TotalTime"] = collect(datetime2unix(BATCH_T0) .+ 40.0 .* (0:n_records-1))
        f["frequency_Hz"] = BATCH_NU .* SPEED_OF_LIGHT_CM_PER_S
        f["WaveToFitROI_1600a"] = intensity
    end
    return path
end

function batch_test_config(; kwargs...)
    defaults = (experiment = "synthetic-test",
                reader = DCSReaderConfig(path_length_cm = 195017.0, pressure_scale = 1.0),
                noise = ConstantNoise(1e-6),
                averaging_period = Minute(2),
                absorbers = BATCH_ABSORBERS,
                retrieval = RetrievalConfig(windows_cm = [(6076.5, 6077.5)],
                                            baseline_coefficients = [3],
                                            fine_grid_spacing_cm = 0.002,
                                            solver = GaussNewton(max_iterations = 10)),
                x_prior = ComponentVector(CH4 = 1.9e-6 * BATCH_N_AIR,
                                          baseline = (window_1 = zeros(3),)),
                prior_sigma = ComponentVector(CH4 = 1e-6 * BATCH_N_AIR,
                                              baseline = (window_1 = fill(10.0, 3),)))
    return BatchRetrievalConfig(; merge(defaults, values(kwargs))...)
end

@testset "BatchRetrievalConfig validation names the field" begin
    mktempdir() do dir
        @test batch_test_config(data_directory = dir, output_root = dir) isa BatchRetrievalConfig
        @test_throws r"data_directory" batch_test_config(data_directory = joinpath(dir, "missing"))
        @test_throws r"data_files" batch_test_config(data_files = [joinpath(dir, "missing.h5")])
        @test_throws r"data_files, data_directory" batch_test_config()
        @test_throws r"averaging_period" batch_test_config(data_directory = dir,
                                                           averaging_period = Minute(-15))
        @test_throws r"averaging_period" batch_test_config(data_directory = dir,
                                                           averaging_period = Minute(0))
        # RetrievalConfig's keyword constructor rejects empty windows itself; the batch
        # configuration checks again for configurations built positionally
        empty_windows = RetrievalConfig(Tuple{Float64,Float64}[], Int[], :multiplicative,
                                        LinearSpace(), ColumnDensity(), DCS(), false, false,
                                        Dict{Symbol,Float64}(), nothing,
                                        Tuple{Float64,Float64}[], :deweight, 1e-5,
                                        :from_measurement, GaussNewton())
        @test_throws r"windows_cm" batch_test_config(data_directory = dir, retrieval = empty_windows)
        @test_throws r"prior_sigma" batch_test_config(data_directory = dir,
            prior_sigma = ComponentVector(CH4 = 1.0, baseline = (window_1 = fill(10.0, 2),)))
        @test_throws r"x_prior" batch_test_config(data_directory = dir,
            x_prior = ComponentVector(CH4 = 1.0, baseline = (window_1 = zeros(2),)),
            prior_sigma = ComponentVector(CH4 = 1.0, baseline = (window_1 = fill(10.0, 2),)))
        @test_throws r"retrieve_pressure" batch_test_config(data_directory = dir,
            x_prior = ComponentVector(CH4 = 1.0, pressure_hPa = 850.0, baseline = (window_1 = zeros(3),)),
            prior_sigma = ComponentVector(CH4 = 1.0, pressure_hPa = 1.0, baseline = (window_1 = ones(3),)))
        @test_throws r"parallel" batch_test_config(data_directory = dir, parallel = :threads)
        @test_throws r"experiment" batch_test_config(data_directory = dir, experiment = "a/b")
    end
end

@testset "batch retrieval end to end, serial and with two worker processes" begin
    mktempdir() do dir
        data = mkpath(joinpath(dir, "data"))
        write_batch_dcs_file(joinpath(data, "20160921a.h5"), 2.0e-6; seed = 1)
        write_batch_dcs_file(joinpath(data, "20160921b.h5"), 2.1e-6; seed = 2, nan_record = 8)
        write(joinpath(data, "20160921c.h5"), "not an HDF5 file")    # matches, unreadable
        write(joinpath(data, "notes.txt"), "skipped: does not match the pattern")
        output_root = joinpath(dir, "runs")
        config = batch_test_config(data_directory = data, output_root = output_root)

        @test BR.input_files(config).skipped == [joinpath(data, "notes.txt")]
        summary = @test_logs((:info, r"Skipping file"), (:warn, r"Could not read file"),
                             (:warn, r"Retrieval failed"), match_mode = :any, BR.run(config))
        @test summary isa BatchRetrievalSummary

        # run folder, configuration copy and manifest
        folder = summary.run_folder
        @test isdir(folder) && dirname(folder) == output_root
        @test occursin(Regex("^" * Dates.format(today(), "yyyy-mm-dd") * "_synthetic-test_"),
                       basename(folder))
        @test isfile(joinpath(folder, "configuration.txt"))
        @test occursin("averaging_period", read(joinpath(folder, "configuration.txt"), String))
        manifest = TOML.parsefile(joinpath(folder, "manifest.toml"))
        @test manifest["n_files_processed"] == 3
        @test manifest["n_files_skipped"] == 1
        @test manifest["n_file_failures"] == 1
        @test manifest["n_measurements"] == 6
        @test manifest["n_converged"] + manifest["n_max_iterations"] == 5
        @test manifest["n_failed"] == 1
        @test length(manifest["result_files"]) == 3
        @test haskey(manifest, "git_hash") && haskey(manifest, "end_time")

        # skipped file, unreadable file, failed measurement recorded without stopping
        @test summary.files_skipped == [joinpath(data, "notes.txt")]
        @test only(summary.file_failures).file == joinpath(data, "20160921c.h5")
        counts = BR.outcome_counts(summary)
        @test counts.total == 6 && counts.failed == 1 && counts.exceptions == 1
        failed = only(o for o in summary.outcomes if o.status === :Failed)
        @test basename(failed.source_file) == "20160921b.h5" && failed.index == 3
        @test failed.time == BATCH_T0 + Minute(4)
        @test occursin("pressure_hPa", failed.message)

        text = report(summary)
        @test occursin("converged=5", text) && occursin("failed=1", text)
        @test occursin("chi2_obs_per_dof", text) && occursin("dfs_total", text)
        @test all(isascii, text)

        # one results file per input; results load back
        @test length(summary.result_files) == 3
        @test all(isfile, summary.result_files)
        a = load_results(joinpath(folder, "results", "20160921a_results.jld2"))
        b = load_results(joinpath(folder, "results", "20160921b_results.jld2"))
        c = load_results(joinpath(folder, "results", "20160921c_results.jld2"))
        @test length(a.results) == 3 && length(b.results) == 2 && isempty(c.results)
        @test !isempty(c.attributes["file_error"])
        @test only(b.attributes["failures"])["index"] == 3
        @test [r.time for r in a.results] == BATCH_T0 .+ Minute.([0, 2, 4])
        @test all(r -> r.metadata["n_averaged"] == 3, a.results)
        for (loaded, truth) in ((a, 2.0e-6), (b, 2.1e-6))
            for r in loaded.results
                @test r.status == Converged
                sigma = sqrt(r.posterior_covariance[1, 1])
                @test abs(r.x.CH4 - truth * BATCH_N_AIR) < 4 * sigma
                @test 0.5 < r.chi2_obs_per_dof < 1.6
            end
        end

        # the same run on two worker processes gives identical results
        new_workers = addprocs(2; exeflags = "--project=$(Base.active_project())")
        try
            Distributed.remotecall_eval(Main, new_workers, :(using SpectralFits))   # @everywhere
            @test nworkers() == 2
            parallel = with_logger(NullLogger()) do
                BR.run(batch_test_config(data_directory = data, output_root = output_root,
                                         parallel = :processes))
            end
            @test parallel.run_folder != folder
            @test [(o.status, o.index, o.time) for o in parallel.outcomes] ==
                  [(o.status, o.index, o.time) for o in summary.outcomes]
            for (serial_file, parallel_file) in zip(summary.result_files, parallel.result_files)
                s, p = load_results(serial_file), load_results(parallel_file)
                @test length(s.results) == length(p.results)
                for (rs, rp) in zip(s.results, p.results)
                    @test rs.x == rp.x
                    @test rs.posterior_covariance == rp.posterior_covariance
                    @test rs.chi2_obs_per_dof == rp.chi2_obs_per_dof
                end
            end
        finally
            rmprocs(new_workers)
        end
    end
end
