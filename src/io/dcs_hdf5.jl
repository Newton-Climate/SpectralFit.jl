# DCS HDF5 reader (section 3.13, phase 8). Replaces v0.1 `read_DCS_data`: the pressure scale
# factor, path length and field names are configuration, not constants (issue 9: the path
# length 195017 is in cm, a 1.95 km round trip).

"""
    DCSFieldLayout(; pressure, temperature, time, frequency = "frequency_Hz",
                   intensity = "WaveToFitROI_1600a")

Names of the HDF5 datasets in a DCS file:

- `pressure`: pressure in hPa (mbar), one value per record.
- `temperature`: temperature in degrees C, one value per record.
- `time`: machine time in seconds since `DCSReaderConfig.time_epoch`, one value per record.
- `frequency`: optical frequency grid in Hz.
- `intensity`: transmitted intensity, a matrix with one record per row
  (`n_records x n_frequencies`, as v0.1 read it); the transpose is accepted.

# Example
```julia
DCSFieldLayout(pressure = "Pres_CS", temperature = "Temp_CS", time = "TotalTime")
```
"""
struct DCSFieldLayout
    pressure::String
    temperature::String
    time::String
    frequency::String
    intensity::String
end
DCSFieldLayout(; pressure, temperature, time, frequency = "frequency_Hz",
               intensity = "WaveToFitROI_1600a") =
    DCSFieldLayout(pressure, temperature, time, frequency, intensity)

# Pressure and time names of the three layouts known from v0.1, in detection order.
const DCS_PRESSURE_TIME_LAYOUTS = (("pressure_evolution", "totaltime_priorfit"),
                                   ("pressure_evolution", "TotalTime"),
                                   ("Pres_CS", "TotalTime"))
# Temperature names in v0.1 priority order (the first present wins).
const DCS_TEMPERATURE_FIELDS = ("temperature_fitted", "temperature_fitted_l", "Temp_CS",
                                "temperature_evolution")

"""Zero of LabVIEW machine time, the default `DCSReaderConfig` time epoch."""
const LABVIEW_EPOCH = DateTime(1904, 1, 1)

"""
    DCSReaderConfig(; path_length_cm = 195017.0, pressure_scale = 1.0,
                    field_layout = nothing, noise_model = ShotNoise(),
                    invalid_records = :skip, time_epoch = LABVIEW_EPOCH)

Settings for `read_dcs_measurements`.

- `path_length_cm`: optical path length in cm stored in each measurement's conditions. The
  default is the v0.1 DCS A round trip (195017 cm = 1.95 km; v0.1 mislabelled it metres).
- `pressure_scale`: dimensionless factor applied to the recorded pressure (default 1). v0.1
  hard-coded 0.9938, a correction for the 2016 campaign only; pass it for those files.
- `field_layout`: a `DCSFieldLayout`, or `nothing` to detect one of the three known layouts:
  `pressure_evolution` + `totaltime_priorfit`, `pressure_evolution` + `TotalTime`, or
  `Pres_CS` + `TotalTime`; temperature is the first present of `temperature_fitted`,
  `temperature_fitted_l`, `Temp_CS`, `temperature_evolution`.
- `noise_model`: an `AbstractNoiseModel` applied to the full-spectrum records, because a
  `Measurement` needs a positive noise variance. The default `ShotNoise()` reproduces v0.1
  (`calc_DCS_noise`, flat window 6255.1-6255.4 cm-1, needs at least two records). Batch
  apps usually replace it with `ResidualNoise` near the retrieval window.
- `invalid_records`: what to do with a record whose intensity is not finite or whose noise
  variance is not positive and finite (for example an all-zero record). `:skip` (default)
  leaves it out and logs a warning with the record indices; `:error` throws.
- `time_epoch`: the `DateTime` at which the machine-time field is zero. The default is the
  LabVIEW epoch `DateTime(1904, 1, 1)` used by the DCS acquisition software (the 20160921
  MERLIN file is one). v0.1 read these times as unix seconds, so its timestamps are 66 years
  late; pass `time_epoch = DateTime(1970, 1, 1)` for files that store unix time.

# Example
```julia
config = DCSReaderConfig(path_length_cm = 196367.0,     # DCS B
                         noise_model = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60)]))
```
"""
struct DCSReaderConfig{N<:AbstractNoiseModel}
    path_length_cm::Float64
    pressure_scale::Float64
    field_layout::Union{Nothing,DCSFieldLayout}
    noise_model::N
    invalid_records::Symbol
    time_epoch::DateTime
end
function DCSReaderConfig(; path_length_cm::Real = 195017.0, pressure_scale::Real = 1.0,
                         field_layout::Union{Nothing,DCSFieldLayout} = nothing,
                         noise_model::AbstractNoiseModel = ShotNoise(),
                         invalid_records::Symbol = :skip,
                         time_epoch::DateTime = LABVIEW_EPOCH)
    path_length_cm > 0 || throw(ArgumentError("path_length_cm must be positive"))
    pressure_scale > 0 || throw(ArgumentError("pressure_scale must be positive"))
    invalid_records in (:skip, :error) ||
        throw(ArgumentError("invalid_records must be :skip or :error, got :$invalid_records"))
    return DCSReaderConfig(Float64(path_length_cm), Float64(pressure_scale), field_layout,
                           noise_model, invalid_records, time_epoch)
end

"""
    detect_dcs_field_layout(file::HDF5.File) -> DCSFieldLayout

Find which of the three known DCS layouts an open HDF5 file uses (see `DCSReaderConfig`).
Throws an `ArgumentError` naming the datasets present when none matches.
"""
function detect_dcs_field_layout(file::HDF5.File)
    pt = findfirst(((p, t),) -> haskey(file, p) && haskey(file, t), DCS_PRESSURE_TIME_LAYOUTS)
    temperature = findfirst(name -> haskey(file, name), DCS_TEMPERATURE_FIELDS)
    if pt === nothing || temperature === nothing
        throw(ArgumentError("unrecognised DCS HDF5 layout; datasets present: " *
                            join(keys(file), ", ")))
    end
    pressure, time = DCS_PRESSURE_TIME_LAYOUTS[pt]
    return DCSFieldLayout(pressure = pressure, temperature = DCS_TEMPERATURE_FIELDS[temperature],
                          time = time)
end

"""
    read_dcs_measurements(path, reader_config = DCSReaderConfig()) -> Vector{Measurement}

Read a dual-comb HDF5 file into one `Measurement` per record:

- `coordinate`: wavenumber in cm-1, the frequency grid in Hz divided by
  `SPEED_OF_LIGHT_CM_PER_S`.
- `y`: the record's intensity (instrument units).
- `noise_variance`: from `reader_config.noise_model` applied to all full-spectrum records.
- `time`: the machine time (seconds since `time_epoch`) as a `DateTime` (millisecond
  resolution).
- `conditions = (pressure_hPa, temperature_K, path_length_cm)`: pressure times
  `pressure_scale`, temperature in degrees C plus 273.15, the configured path length.
- `metadata = (source_file, machine_time, record_index, n_averaged = 1)`.

The public form `read_measurements(DCS, path; reader_config)` forwards here.

# Example
```julia
ms = read_dcs_measurements("data/20160921.h5", DCSReaderConfig())
ms[1].conditions.pressure_hPa
```
"""
function read_dcs_measurements(path::AbstractString, reader_config::DCSReaderConfig = DCSReaderConfig())
    raw = h5open(path, "r") do file
        layout = reader_config.field_layout === nothing ? detect_dcs_field_layout(file) :
                 reader_config.field_layout
        (pressure = vec(read(file, layout.pressure)),
         temperature = vec(read(file, layout.temperature)),
         machine_time = vec(read(file, layout.time)),
         frequency = vec(read(file, layout.frequency)),
         intensity = read(file, layout.intensity))
    end
    n_records = length(raw.pressure)
    length(raw.temperature) == n_records && length(raw.machine_time) == n_records ||
        throw(DimensionMismatch("$path: pressure, temperature and time have different lengths"))
    intensity = _records_by_row(raw.intensity, n_records, length(raw.frequency), path)

    wavenumber = Float64.(raw.frequency) ./ SPEED_OF_LIGHT_CM_PER_S
    pressure_hPa = reader_config.pressure_scale .* Float64.(raw.pressure)
    temperature_K = Float64.(raw.temperature) .+ 273.15
    variances = record_noise_variances(reader_config.noise_model, wavenumber, intensity)
    valid = [v > 0 && isfinite(v) && all(isfinite, view(intensity, i, :))
             for (i, v) in enumerate(variances)]
    if !all(valid)
        bad = findall(!, valid)
        reader_config.invalid_records === :error && throw(ArgumentError(
            "$path: records $bad have non-finite intensity or a non-positive or non-finite " *
            "noise variance"))
        @warn "skipping $(length(bad)) of $n_records records with non-finite intensity or invalid noise variance" path records = bad
    end

    source = String(path)
    return [Measurement(y = intensity[i, :], coordinate = wavenumber,
                        noise_variance = variances[i],
                        time = machine_time_to_datetime(raw.machine_time[i], reader_config.time_epoch),
                        conditions = (pressure_hPa = pressure_hPa[i],
                                      temperature_K = temperature_K[i],
                                      path_length_cm = reader_config.path_length_cm),
                        metadata = (source_file = source,
                                    machine_time = Float64(raw.machine_time[i]),
                                    record_index = i, n_averaged = 1))
            for i in 1:n_records if valid[i]]
end

"""Seconds since `epoch` as a `DateTime`, to the millisecond."""
machine_time_to_datetime(seconds::Real, epoch::DateTime) =
    epoch + Millisecond(round(Int64, 1000 * Float64(seconds)))

# Intensity as n_records x n_frequencies, accepting either orientation (records-by-row wins
# when the matrix is square, as in v0.1).
function _records_by_row(intensity::AbstractVecOrMat, n_records, n_frequencies, path)
    M = intensity isa AbstractVector ? reshape(intensity, 1, :) : intensity
    size(M) == (n_records, n_frequencies) && return Float64.(M)
    size(M) == (n_frequencies, n_records) && return permutedims(Float64.(M))
    throw(DimensionMismatch("$path: intensity has size $(size(M)), expected " *
                            "($n_records, $n_frequencies) or its transpose"))
end

export LABVIEW_EPOCH, DCSFieldLayout, DCSReaderConfig, detect_dcs_field_layout, read_dcs_measurements

"""
    read_measurements(DCS, path; reader_config = DCSReaderConfig()) -> Vector{Measurement}

Read every record of a DCS HDF5 file (see `read_dcs_measurements`).

```julia
measurements = read_measurements(DCS, "20160921.h5"; reader_config = DCSReaderConfig())
```
"""
read_measurements(::Type{<:DCS}, path::AbstractString; reader_config::DCSReaderConfig = DCSReaderConfig()) =
    read_dcs_measurements(path, reader_config)

export read_measurements
