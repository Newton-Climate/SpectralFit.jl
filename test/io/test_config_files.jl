# Configuration files: extends, variants, path resolution; conditions CSV; run folders.
import TOML

@testset "read_config follows extends, resolves paths and expands variants" begin
    mktempdir() do dir
        mkpath(joinpath(dir, "base"))
        write(joinpath(dir, "base", "campaign.toml"), """
            app = "batch_retrieval"
            experiment = "base"
            output_root = "runs"
            [reader]
            pressure_scale = 0.9938
            path_length_cm = 195017.0
            [retrieval]
            baseline_coefficients = 100
            """)
        write(joinpath(dir, "hit08.toml"), """
            extends = "base/campaign.toml"
            experiment = "hit08"
            [absorbers.CH4]
            file = "spectra/ch4.jld2"
            [retrieval]
            retrieve_pressure = true
            [[variants]]
            name = "ch4"
            retrieval.window_cm = [6050.0, 6115.0]
            [[variants]]
            name = "co2"
            retrieval = { window_cm = [6180.0, 6260.0], baseline_coefficients = 50 }
            """)
        configs = SF.read_config(joinpath(dir, "hit08.toml"))
        @test length(configs) == 2
        a, b = configs
        @test a["experiment"] == "hit08_ch4" && b["experiment"] == "hit08_co2"
        @test a["variant"] == "ch4"
        @test a["reader"]["pressure_scale"] == 0.9938                         # inherited
        @test a["output_root"] == normpath(joinpath(dir, "base", "runs"))      # relative to base file
        @test a["absorbers"]["CH4"]["file"] == normpath(joinpath(dir, "spectra", "ch4.jld2"))
        @test a["retrieval"]["window_cm"] == [6050.0, 6115.0]
        @test a["retrieval"]["retrieve_pressure"] && a["retrieval"]["baseline_coefficients"] == 100
        @test b["retrieval"]["baseline_coefficients"] == 50
        @test !haskey(a, "variants") && !haskey(a, "extends")
        @test [c["variant"] for c in SF.read_config(joinpath(dir, "hit08.toml"); variants = ["co2"])] == ["co2"]
        @test_throws r"no variants named" SF.read_config(joinpath(dir, "hit08.toml"); variants = ["xx"])
        write(joinpath(dir, "loop.toml"), "extends = \"loop.toml\"\nexperiment = \"x\"\n")
        @test_throws r"loop" SF.read_config(joinpath(dir, "loop.toml"))
        write(joinpath(dir, "dup.toml"), "experiment = \"x\"\n[[variants]]\nname = \"a\"\n[[variants]]\nname = \"a\"\n")
        @test_throws r"twice" SF.read_config(joinpath(dir, "dup.toml"))
        SF.write_config(joinpath(dir, "copy.toml"), a)
        @test TOML.parsefile(joinpath(dir, "copy.toml"))["experiment"] == "hit08_ch4"
    end
end

@testset "spec parsers name the offending key" begin
    @test SF.solver_from_config(Dict("name" => "levenberg_marquardt", "initial_damping" => 40.0)).damping.initial == 40.0
    @test SF.solver_from_config(Dict{String,Any}()) isa GaussNewton
    @test_throws r"unknown keys" SF.solver_from_config(Dict("name" => "gauss_newton", "maxiter" => 3))
    @test SF.noise_model_from_config(Dict("model" => "residual", "flat_windows_cm" => [[1.0, 2.0]])) isa ResidualNoise
    @test_throws r"model must be" SF.noise_model_from_config(Dict("model" => "white"))
    r = SF.retrieval_config_from_config(Dict("window_cm" => [6076.0, 6078.0], "baseline_coefficients" => 5,
                                             "masked_windows_cm" => [[6076.5, 6076.6]], "fine_grid_margin_cm" => 0.05,
                                             "h2o_broadening" => Dict("CH4" => 0.34)); solver = GaussNewton())
    @test r.windows_cm == [(6076.0, 6078.0)] && r.baseline_coefficients == [5]
    @test r.masked_windows_cm == [(6076.5, 6076.6)] && r.fine_grid_margin_cm == 0.05
    @test r.h2o_broadening == Dict(:CH4 => 0.34)
    @test_throws r"exactly one of window_cm" SF.retrieval_config_from_config(Dict("baseline_coefficients" => 5); solver = GaussNewton())
    a = SF.absorber_from_config(:CH4, Dict("file" => FIXTURE_PAR, "molecule" => 6, "range_cm" => [6076.0, 6080.0]))
    @test a.model isa HitranVoigt && a.name == :CH4
    @test_throws r"missing key \"molecule\"" SF.absorber_from_config(:CH4, Dict("file" => FIXTURE_PAR, "range_cm" => [6076.0, 6080.0]))
end

@testset "read_conditions_csv" begin
    mktempdir() do dir
        path = joinpath(dir, "tower.csv")
        write(path, "# tower\ntime,pressure_hPa,temperature_K\n2016-09-21T09:14:19,821.8,295.3\n2016-09-21T09:00:00,820.0,294.0\n")
        t = SF.read_conditions_csv(path; max_offset = Second(30))
        @test t.times == [DateTime(2016, 9, 21, 9), DateTime(2016, 9, 21, 9, 14, 19)]     # sorted
        @test t.values.pressure_hPa == [820.0, 821.8]
        @test t.max_offset == Millisecond(30_000)
        write(path, "pressure_hPa\n1.0\n")
        @test_throws r"no \"time\" column" SF.read_conditions_csv(path)
    end
end

@testset "run folders" begin
    mktempdir() do dir
        a = SF.create_run_folder(dir, "exp"; git_hash = "abc1234")
        sleep(1.1)                                            # distinct modification times
        b = SF.create_run_folder(dir, "exp"; git_hash = "abc1234")
        c = SF.create_run_folder(dir, "exp_lm"; git_hash = "abc1234")
        @test isdir(joinpath(a, "results")) && endswith(b, "_2")
        @test SF.find_run(dir, "exp"; require_results = false) == b      # newest, not exp_lm
        @test SF.find_run(dir, "exp_lm"; require_results = false) == c
        @test SF.run_experiments(dir) == ["exp", "exp_lm"]
        @test_throws r"no run folder" SF.find_run(dir, "other")
        @test_throws r"experiment" SF.create_run_folder(dir, "a/b")
    end
end

@testset "variants can replace or remove tables" begin
    base = Dict{String,Any}("absorbers" => Dict{String,Any}(
        "CH4" => Dict{String,Any}("file" => "ch4.par", "molecule" => 6, "range_cm" => [1.0, 2.0]),
        "CO2" => Dict{String,Any}("file" => "co2.par", "molecule" => 2)))
    m = SF.merge_config(base, Dict{String,Any}("absorbers" => Dict{String,Any}(
        "CH4" => Dict{String,Any}("_replace" => true, "file" => "ch4.jld2"),
        "CO2" => Dict{String,Any}("_remove" => true))))
    @test m["absorbers"] == Dict("CH4" => Dict("file" => "ch4.jld2"))
    @test base["absorbers"]["CH4"]["molecule"] == 6                         # base unchanged
    @test SF.merge_config(Dict{String,Any}(), Dict{String,Any}("a" => Dict{String,Any}("_replace" => true, "b" => 1))) ==
          Dict("a" => Dict("b" => 1))                                       # markers never leak
end

@testset "find_run passes over runs without results" begin
    mktempdir() do dir
        good = SF.create_run_folder(dir, "exp"; git_hash = "abc1234")
        write(joinpath(good, "results", "x_results.jld2"), "x")
        sleep(1.1)
        SF.create_run_folder(dir, "exp"; git_hash = "abc1234")            # newer, aborted
        @test SF.find_run(dir, "exp") == good
        @test SF.find_run(dir, "exp"; require_results = false) != good
    end
end
