# Phase 8: DCS HDF5 reader with the three known field layouts (issue 9 path length).
using HDF5

const DCS_FIXTURE_LAYOUTS = (
    (pressure = "pressure_evolution", temperature = "temperature_fitted_l", time = "totaltime_priorfit"),
    (pressure = "pressure_evolution", temperature = "temperature_fitted", time = "TotalTime"),
    (pressure = "Pres_CS", temperature = "Temp_CS", time = "TotalTime"),
)

"""Write a small DCS file with the given field names; returns the values written."""
function write_dcs_fixture(path, names; n_records = 4, n_frequencies = 600, seed = 1)
    rng = MersenneTwister(seed)
    nu = collect(range(6255.0, 6255.5; length = n_frequencies))
    frequency_Hz = nu .* SPEED_OF_LIGHT_CM_PER_S
    pressure = [830.0, 831.0, 832.5, 829.0][1:n_records]
    temperature_C = [20.0, 21.5, 19.0, 22.0][1:n_records]
    machine_time = 1.4744736e9 .+ 1.5 .* (0:n_records-1)
    intensity = reduce(vcat, ((level .+ 1e-3 .* randn(rng, n_frequencies))'
                              for level in (1.0, 1.1, 0.9, 1.05)[1:n_records]))
    h5open(path, "w") do f
        f[names.pressure] = pressure
        f[names.temperature] = temperature_C
        f[names.time] = collect(machine_time)
        f["frequency_Hz"] = frequency_Hz
        f["WaveToFitROI_1600a"] = intensity
    end
    return (; nu, pressure, temperature_C, machine_time, intensity)
end

@testset "DCS reader: three layouts, units, conditions" begin
    mktempdir() do dir
        for (k, names) in enumerate(DCS_FIXTURE_LAYOUTS)
            path = joinpath(dir, "layout_$k.h5")
            truth = write_dcs_fixture(path, names)
            ms = read_dcs_measurements(path, DCSReaderConfig())
            @test length(ms) == 4
            @test ms[1].coordinate ≈ truth.nu rtol = 1e-12                 # Hz -> cm-1
            @test [m.conditions.pressure_hPa for m in ms] == truth.pressure          # scale 1
            @test [m.conditions.temperature_K for m in ms] ≈ truth.temperature_C .+ 273.15
            @test all(m -> m.conditions.path_length_cm == 195017.0, ms)
            @test ms[3].y == truth.intensity[3, :]
            @test ms[1].time == LABVIEW_EPOCH + Millisecond(round(Int64, 1000 * truth.machine_time[1]))
            @test ms[2].metadata.machine_time == truth.machine_time[2]
            @test ms[2].metadata.source_file == path
            @test ms[2].metadata.n_averaged == 1
            @test all(m -> all(>(0), m.noise_variance), ms)               # default ShotNoise
            layout = h5open(detect_dcs_field_layout, path)
            @test layout.pressure == names.pressure
            @test layout.temperature == names.temperature
            @test layout.time == names.time
        end
    end
end

@testset "DCS reader: configuration options" begin
    mktempdir() do dir
        path = joinpath(dir, "custom.h5")
        names = (pressure = "p_custom", temperature = "t_custom", time = "time_custom")
        truth = write_dcs_fixture(path, names)
        @test_throws ArgumentError read_dcs_measurements(path, DCSReaderConfig())  # unknown layout
        config = DCSReaderConfig(path_length_cm = 196367.0, pressure_scale = 0.9938,
                                 field_layout = DCSFieldLayout(; names...),
                                 noise_model = ConstantNoise(1e-6))
        ms = read_dcs_measurements(path, config)
        @test [m.conditions.pressure_hPa for m in ms] ≈ 0.9938 .* truth.pressure
        @test all(m -> m.conditions.path_length_cm == 196367.0, ms)
        @test all(m -> m.noise_variance == fill(1e-6, length(truth.nu)), ms)
        # residual noise model recovers the 1e-3 white noise of the fixture
        res = DCSReaderConfig(field_layout = DCSFieldLayout(; names...),
                              noise_model = ResidualNoise(flat_windows_cm = [(6255.0, 6255.5)]))
        v = [m.noise_variance[1] for m in read_dcs_measurements(path, res)]
        @test all(isapprox.(v, 1e-6; rtol = 0.2))
        # reader output feeds time_average
        avg = time_average(ms, Second(3))
        @test sum(a.metadata.n_averaged for a in avg) == 4
        @test_throws ArgumentError DCSReaderConfig(path_length_cm = -1.0)
    end
end

@testset "DCS reader: invalid records are skipped or rejected" begin
    mktempdir() do dir
        path = joinpath(dir, "bad_record.h5")
        names = DCS_FIXTURE_LAYOUTS[3]
        truth = write_dcs_fixture(path, names)
        bad = copy(truth.intensity); bad[2, 10] = NaN
        h5open(path, "r+") do f
            delete_object(f, "WaveToFitROI_1600a")
            f["WaveToFitROI_1600a"] = bad
        end
        config = DCSReaderConfig(noise_model = ConstantNoise(1e-6))
        ms = @test_logs (:warn, r"skipping 1 of 4 records") read_dcs_measurements(path, config)
        @test [m.metadata.record_index for m in ms] == [1, 3, 4]
        strict = DCSReaderConfig(noise_model = ConstantNoise(1e-6), invalid_records = :error)
        @test_throws r"records \[2\]" read_dcs_measurements(path, strict)
        @test_throws r"invalid_records" DCSReaderConfig(invalid_records = :drop)
    end
end

@testset "DCS reader: machine-time epoch" begin
    mktempdir() do dir
        path = joinpath(dir, "epoch.h5")
        truth = write_dcs_fixture(path, DCS_FIXTURE_LAYOUTS[3])
        labview = read_dcs_measurements(path, DCSReaderConfig(noise_model = ConstantNoise(1e-6)))
        @test labview[1].time == DateTime(1904, 1, 1) + Millisecond(round(Int64, 1000 * truth.machine_time[1]))
        unix = read_dcs_measurements(path, DCSReaderConfig(noise_model = ConstantNoise(1e-6),
                                                           time_epoch = DateTime(1970, 1, 1)))
        @test unix[1].time == unix2datetime(truth.machine_time[1])
    end
end
