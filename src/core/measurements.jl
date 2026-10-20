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
