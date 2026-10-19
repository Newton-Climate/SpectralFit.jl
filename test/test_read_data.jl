# Unit tests for src/read_data.jl

"""Small hand-built dataset with samples at the given minute offsets."""
function toy_dataset(minutes; start=DateTime(2020, 1, 1, 0, 0, 30))
    n = length(minutes)
    times = [start + Minute(m) for m in minutes]
    return SF.FrequencyCombDataset(filename="toy", intensity=Float64[i + j for i in 1:n, j in 1:3],
                                   grid=[1.0, 2.0, 3.0], temperature=collect(280.0 .+ (1:n)),
                                   pressure=collect(800.0 .+ (1:n)), time=times, pathlength=1.0,
                                   num_averaged_measurements=ones(Int, n), averaging_window=Second(1),
                                   machine_time=datetime2unix.(times), σ²=fill(4.0, n),
                                   vcd=ones(n))
end

@testset "read_DCS_data" begin
    mktempdir() do dir
        path = joinpath(dir, "20160921.h5")
        truth = write_synthetic_dcs(path; n_times=30)
        ds = read_DCS_data(path)

        @test ds isa FrequencyCombDataset{Float64}
        @test ds.filename == path
        @test size(ds.intensity) == (30, length(truth.grid_wn))
        @test ds.grid ≈ truth.grid_wn
        @test all(ds.pressure .≈ P_SURF)
        @test all(ds.temperature .≈ T_SURF)
        @test ds.time == unix2datetime.(truth.machine_time)
        @test ds.pathlength == 195017.0
        @test length(ds.σ²) == 30
        # the flat region carries white noise σ = 1e-4 on a unit baseline
        @test mean(ds.σ²) ≈ 1e-8 rtol = 0.3
        @test ds.vcd ≈ SF.calc_vcd.(ds.pressure, ds.temperature, ds.pathlength)
    end
end

@testset "take_time_average!" begin
    ds = toy_dataset(0:29)
    raw_intensity = copy(ds.intensity)
    take_time_average!(ds; δt=Minute(15))

    @test length(ds.time) == 2
    @test ds.num_averaged_measurements == [15, 15]
    @test ds.intensity[1, :] ≈ vec(mean(raw_intensity[1:15, :], dims=1))
    @test ds.intensity[2, :] ≈ vec(mean(raw_intensity[16:30, :], dims=1))
    @test ds.pressure ≈ [mean(801.0:815.0), mean(816.0:830.0)]
    @test ds.σ² ≈ fill(4.0 / 15, 2)              # noise of the mean
    @test ds.averaging_window == Minute(15)
    @test ds.time[1] == DateTime(2020, 1, 1, 0, 0, 30)

    # empty averaging windows are dropped instead of leaving uninitialized rows
    gappy = toy_dataset([0, 1, 2, 31, 32, 33])
    take_time_average!(gappy; δt=Minute(15))
    @test gappy.num_averaged_measurements == [3, 3]
    @test size(gappy.intensity, 1) == 2
    @test all(isfinite, gappy.intensity)
    @test length(gappy.time) == length(gappy.σ²) == length(gappy.vcd) == 2
end

@testset "get_measurement" begin
    ds = toy_dataset(0:4)
    ds.grid = [6000.0, 6001.0, 6002.0]
    m = get_measurement(2, ds, 6000.5, 6003.0)
    @test m isa FrequencyCombMeasurement
    @test m.grid == [6001.0, 6002.0]
    @test m.intensity == ds.intensity[2, 2:3]
    @test m.pressure == ds.pressure[2]
    @test m.time == ds.time[2]
    @test m.σ² == ds.σ²[2]

    # profile variant replaces p, T with layer profiles and computes the vcd profile
    p, T = [900.0, 700.0], [290.0, 270.0]
    mp = get_measurement(1, ds, 6000.5, 6003.0, p, T)
    @test mp.pressure == p
    @test mp.vcd ≈ make_vcd_profile(p, T)

    all_m = SF.data2measurements(ds)
    @test length(all_m) == 5
end
