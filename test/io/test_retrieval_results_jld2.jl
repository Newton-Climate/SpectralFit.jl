# Phase 8: retrieval results round-trip through save_results / load_results.
using JLD2

"""Two retrieval results on a linear problem with a nested state: one solved, one Failed."""
function results_fixture()
    rng = MersenneTwister(7)
    x_prior = ComponentVector(CH4 = 1.0, H2O = 2.0, baseline = (window_1 = [0.5, -0.3],))
    n_obs = 25
    K = randn(rng, n_obs, 4)
    Se_inv = Diagonal(fill(400.0, n_obs))
    B = randn(rng, 4, 4)
    Sa_inv = B * B' / 4 + 0.5I                     # dense prior precision
    y = K * [1.3, 2.1, 0.2, -0.1] + 0.05 * randn(rng, n_obs)
    problem = RetrievalProblem(x -> K * x, y, Se_inv, x_prior, Sa_inv)
    solved = solve(problem, GaussNewton())
    failed = retrieval_result(problem, collect(x_prior), fill(NaN, n_obs, 4), K * collect(x_prior);
                              status = MaxIterations, message = "test", n_iterations = 30,
                              rodgers_d2_per_state = 0.5, cost_history = [3.0, 2.0])
    return [solved, failed]
end

@testset "state_from_labels rebuilds ComponentVectors" begin
    x = ComponentVector(CH4 = 1.0, b = (w1 = [2.0, 3.0], w2 = [4.0]), m = [5.0 7.0; 6.0 8.0])
    y = state_from_labels(collect(x), state_labels(x))
    @test y == x
    @test getaxes(y) == getaxes(x)
    @test y.b.w1 == [2.0, 3.0] && y.m == [5.0 7.0; 6.0 8.0]
    @test_throws ArgumentError state_from_labels([1.0], ["a", "b"])
    @test_throws ArgumentError state_from_labels([1.0, 2.0], ["a[2]", "a[1]"])
end

@testset "save_results / load_results round trip with diagnostics" begin
    results = results_fixture()
    @test results[1].status == Converged
    @test results[2].status == Failed
    times = [DateTime(2016, 9, 21, 0, 0), DateTime(2016, 9, 21, 0, 15)]
    metadata = [(source_file = "a.h5", record_index = 1, n_averaged = 3,
                 averaging_period = Minute(15)),
                (source_file = "a.h5", record_index = 4, n_averaged = 2,
                 averaging_period = Minute(15))]
    mktempdir() do dir
        path = joinpath(dir, "results.jld2")
        @test save_results(path, results; times = times, metadata = metadata,
                           attributes = (source_file = "a.h5",
                                         failures = [(index = 3, message = "boom")])) == path
        loaded = load_results(path)
        @test length(loaded.results) == 2
        @test loaded.attributes["source_file"] == "a.h5"
        @test loaded.attributes["failures"][1]["message"] == "boom"
        for (r, l, t) in zip(results, loaded.results, times)
            as_array(v) = v isa AbstractArray ? collect(v) : v
            for name in fieldnames(RetrievalResult)
                @test isequal(as_array(getfield(r, name)), as_array(getproperty(l, name)))
            end
            @test l.x isa ComponentVector && getaxes(l.x) == getaxes(r.x)
            @test l.x.baseline.window_1 == r.x.baseline.window_1
            @test isequal(l.dfs_per_element.CH4, r.dfs_per_element.CH4)
            @test l.status isa RetrievalStatus
            @test l.time == t
            @test l.state_labels == state_labels(r.x)
        end
        @test loaded.results[1].obs_precision isa Diagonal
        @test loaded.results[1].prior_precision isa Matrix
        @test loaded.results[2].message == results[2].message
        @test loaded.results[1].metadata["averaging_period"] == "15 minutes"
        @test loaded.results[1].metadata["n_averaged"] == 3

        # only plain data in the file: no SpectralFits or ComponentArrays types
        raw = jldopen(path, "r") do f
            f["results"]
        end
        @test raw isa Vector{Dict{String,Any}}
        plain(v) = v isa Union{Real,String} || (v isa Array && (eltype(v) <: Union{Real,String} ||
                                                                 all(plain, v))) ||
                   (v isa Dict{String,Any} && all(plain, values(v)))
        @test all(plain, raw)

        # empty results and no times
        empty_path = joinpath(dir, "empty.jld2")
        save_results(empty_path, RetrievalResult[])
        @test isempty(load_results(empty_path).results)
        save_results(empty_path, results[1:1])
        @test load_results(empty_path).results[1].time === nothing
        @test_throws ArgumentError save_results(empty_path, results; times = times[1:1])
        jldsave(joinpath(dir, "other.jld2"); a = 1)
        @test_throws ArgumentError load_results(joinpath(dir, "other.jld2"))
    end
end
