# Phase 8: time_average (issue 5).

function make_time_series(times; nv = 0.04)
    nu = collect(6000.0:1.0:6004.0)
    return [Measurement(y = fill(Float64(i), length(nu)), coordinate = nu,
                        noise_variance = nv * i, time = t,
                        conditions = (pressure_hPa = 800.0 + i, temperature_K = 290.0 + i,
                                      path_length_cm = 195017.0, site = "boulder"),
                        metadata = (source_file = "f.h5", machine_time = Float64(i),
                                    n_averaged = 1))
            for (i, t) in enumerate(times)]
end

@testset "half-open bins, edge sample counted once, single bins kept" begin
    t0 = DateTime(2016, 9, 21, 12, 0, 0)
    times = [t0 + Minute(1), t0 + Minute(7), t0 + Minute(15),   # bin [12:00,12:15): 1, 2
             t0 + Minute(20),                                    # bin [12:15,12:30): 3, 4
             t0 + Minute(47)]                                    # bin [12:45,13:00): 5 alone
    ms = make_time_series(times)
    avg = time_average(ms, Minute(15))
    @test length(avg) == 3
    @test [a.time for a in avg] == [t0, t0 + Minute(15), t0 + Minute(45)]
    @test [a.metadata.n_averaged for a in avg] == [2, 2, 1]
    @test sum(a.metadata.n_averaged for a in avg) == length(ms)      # each sample once
    @test avg[2].y == fill(3.5, 5)                                   # edge sample in later bin
    @test avg[3].y == ms[5].y                                        # single-sample bin kept
    @test avg[3].noise_variance == ms[5].noise_variance
    # noise variance sum / count^2
    @test avg[1].noise_variance ≈ fill((0.04 * 1 + 0.04 * 2) / 4, 5)
    @test avg[2].noise_variance ≈ fill((0.04 * 3 + 0.04 * 4) / 4, 5)
    # conditions averaged numerically, non-numeric kept
    @test avg[1].conditions.pressure_hPa ≈ 801.5
    @test avg[1].conditions.temperature_K ≈ 291.5
    @test avg[1].conditions.path_length_cm == 195017.0
    @test avg[1].conditions.site == "boulder"
    # metadata: first sample's values, period recorded
    @test avg[2].metadata.machine_time == 3.0
    @test avg[2].metadata.source_file == "f.h5"
    @test avg[1].metadata.averaging_period == Minute(15)
end

@testset "time_average is pure, sorts input, nests counts" begin
    t0 = DateTime(2016, 9, 21, 12, 3, 0)
    times = [t0 + Second(s) for s in (50, 0, 10, 70, 20)]           # unsorted
    ms = make_time_series(times)
    y_before = [copy(m.y) for m in ms]
    avg = time_average(ms, Minute(1))
    @test [m.y for m in ms] == y_before
    @test length(ms) == 5
    @test avg[1].time == DateTime(2016, 9, 21, 12, 3, 0)            # floor of first time
    @test avg[1].metadata.n_averaged == 4
    @test avg[2].metadata.n_averaged == 1
    @test avg[1].metadata.machine_time == 2.0                       # earliest sample
    # averaging an averaged series sums n_averaged
    @test time_average(avg, Hour(1))[1].metadata.n_averaged == 5
    @test avg isa Vector{<:Measurement}
    @test_throws ArgumentError time_average(Measurement[], Minute(1))
    other = Measurement(y = ones(3), coordinate = [1.0, 2.0, 3.0], noise_variance = 1.0,
                        time = t0, conditions = ms[1].conditions)
    @test_throws ArgumentError time_average([ms[1], other], Minute(1))
end
