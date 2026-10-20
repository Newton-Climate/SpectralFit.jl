# Measurements (section 3.9). A measurement is immutable, holds the observation in instrument
# units and no physics settings.

"""
    Measurement(; y, coordinate, noise_variance, time = DateTime(0), conditions = (;), metadata = (;))

One observation vector. `y` is in instrument units (never log-transformed), `coordinate` is
wavenumber in cm-1 for spectrometers or sample time in s for WMS, `noise_variance` is per
observation in instrument units squared (a scalar is expanded), `time` is the start of the
averaging interval, `conditions` a named tuple such as `(pressure_hPa = 850.0,
temperature_K = 290.0, path_length_cm = 195017.0)` and `metadata` a named tuple (number
averaged, source file, machine time).
"""
struct Measurement{FT<:Real, C<:AbstractVector{FT}, A<:NamedTuple, MD<:NamedTuple}
    y::Vector{FT}
    coordinate::C
    noise_variance::Vector{FT}
    time::DateTime
    conditions::A
    metadata::MD
    function Measurement{FT,C,A,MD}(y, coordinate, noise_variance, time, conditions,
                                    metadata) where {FT,C,A,MD}
        n = length(y)
        length(coordinate) == n ||
            throw(DimensionMismatch("coordinate has length $(length(coordinate)), y has length $n"))
        length(noise_variance) == n ||
            throw(DimensionMismatch("noise_variance has length $(length(noise_variance)), y has length $n"))
        all(v -> v > 0, noise_variance) ||
            throw(ArgumentError("noise_variance must be positive"))
        return new{FT,C,A,MD}(y, coordinate, noise_variance, time, conditions, metadata)
    end
end

function Measurement(; y, coordinate, noise_variance, time::DateTime = DateTime(0),
                     conditions::NamedTuple = (;), metadata::NamedTuple = (;))
    FT = float(promote_type(eltype(y), eltype(coordinate), eltype(noise_variance)))
    yv = convert(Vector{FT}, collect(y))
    c = coordinate isa AbstractRange ?
        range(FT(first(coordinate)), FT(last(coordinate)); length = length(coordinate)) :
        convert(Vector{FT}, collect(coordinate))
    nv = noise_variance isa Real ? fill(FT(noise_variance), length(yv)) :
         convert(Vector{FT}, collect(noise_variance))
    return Measurement{FT,typeof(c),typeof(conditions),typeof(metadata)}(yv, c, nv, time,
                                                                         conditions, metadata)
end

"""
    select_window(measurement, minimum_cm, maximum_cm)

New measurement restricted to the open interval `minimum_cm < coordinate < maximum_cm`
(open question 13: a point on a window edge is excluded, as in v0.1).
"""
function select_window(m::Measurement, minimum_cm::Real, maximum_cm::Real)
    minimum_cm < maximum_cm ||
        throw(ArgumentError("select_window: minimum_cm $minimum_cm must be below maximum_cm $maximum_cm"))
    idx = findall(c -> minimum_cm < c < maximum_cm, m.coordinate)
    isempty(idx) && throw(ArgumentError("select_window: no observations in ($minimum_cm, $maximum_cm)"))
    return Measurement(y = m.y[idx], coordinate = m.coordinate[idx],
                       noise_variance = m.noise_variance[idx], time = m.time,
                       conditions = m.conditions, metadata = m.metadata)
end

n_observations(m::Measurement) = length(m.y)

export Measurement, select_window

# ---------------------------------------------------------------- time averaging (issue 5)

"""
    time_average(measurements::AbstractVector{<:Measurement}, period::Dates.Period) -> Vector{Measurement}

Average a measurement series into time bins of length `period`. Pure: the input is not
changed. Fixes issue 5 (v0.1 `take_time_average!` used closed bins and dropped bins with one
sample).

Binning rules:

- Bins are half-open intervals `[start, start + period)`. A measurement belongs to the bin
  whose start is `floor(m.time, period)`, so bin edges are aligned to the `Dates` rounding
  origin (`0000-01-01T00:00:00`, with weeks starting on Monday). For the usual periods
  (seconds, minutes, hours, days) this means bins start on whole clock values: with
  `Minute(15)` they start at :00, :15, :30, :45, and the first bin starts at
  `floor(first time, period)`. A sample exactly on an edge goes into the later bin only.
- Empty bins are omitted; bins with a single sample are kept. Output is sorted by bin start.
- `y` is the mean over the bin; `noise_variance` is `sum(noise_variance) / count^2`
  (variance of the mean of independent samples); `time` is the bin start.
- `conditions`: each field is averaged when its values are numeric; a non-numeric field
  (a string, a symbol) keeps the value of the earliest sample in the bin.
- `metadata`: `n_averaged` is the sum of the inputs' `n_averaged` (1 when absent),
  `averaging_period` is `period`; every other field keeps the value of the earliest sample in
  the bin (for example `machine_time` and `source_file`).

All measurements must share one coordinate axis and the same `conditions` field names.

# Example
```julia
ms = read_dcs_measurements("20160921.h5", DCSReaderConfig())
averaged = time_average(ms, Minute(15))
averaged[1].metadata.n_averaged      # number of records in the first bin
```
"""
function time_average(measurements::AbstractVector{<:Measurement}, period::Dates.Period)
    isempty(measurements) && throw(ArgumentError("time_average: no measurements"))
    Dates.value(period) > 0 || throw(ArgumentError("time_average: period must be positive"))
    reference = first(measurements)
    for m in measurements
        m.coordinate == reference.coordinate ||
            throw(ArgumentError("time_average: measurements do not share one coordinate axis"))
        keys(m.conditions) == keys(reference.conditions) ||
            throw(ArgumentError("time_average: measurements have different condition fields"))
    end
    order = sortperm([m.time for m in measurements])     # stable: ties keep input order
    starts = [floor(measurements[i].time, period) for i in order]
    result = Measurement[]
    i = 1
    while i <= length(order)
        j = i
        while j < length(order) && starts[j+1] == starts[i]
            j += 1
        end
        push!(result, _average_bin(measurements[order[i:j]], starts[i], period))
        i = j + 1
    end
    return [m for m in result]       # concretely typed when every bin has the same type
end

function _average_bin(bin::AbstractVector{<:Measurement}, start::DateTime, period::Dates.Period)
    n = length(bin)
    first_m = first(bin)
    y = sum(m.y for m in bin) ./ n
    noise_variance = sum(m.noise_variance for m in bin) ./ n^2
    conditions = _average_named_tuples([m.conditions for m in bin])
    n_averaged = sum(m -> Int(get(m.metadata, :n_averaged, 1)), bin)
    metadata = merge(first_m.metadata, (n_averaged = n_averaged, averaging_period = period))
    return Measurement(y = y, coordinate = first_m.coordinate, noise_variance = noise_variance,
                       time = start, conditions = conditions, metadata = metadata)
end

function _average_named_tuples(tuples::AbstractVector{<:NamedTuple})
    names = keys(first(tuples))
    values = map(names) do name
        column = [t[name] for t in tuples]
        all(v -> v isa Real, column) ? sum(column) / length(column) : first(column)
    end
    return NamedTuple{names}(values)
end

export time_average
