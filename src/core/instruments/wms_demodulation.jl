# WMS instrument and demodulators (section 3.7 of docs/architecture_plan.md).
#
# Harmonic convention, shared by both demodulators: with reference angle w t (w = 2 pi f) the
# detector signal over a modulation period is written
#     S(t) = DC + sum_n [ X_n cos(n w t) + Y_n sin(n w t) ],
# so X_n = (2/P) int S cos(n w t) dt, Y_n = (2/P) int S sin(n w t) dt and DC = (1/P) int S dt.
# A component R cos(n w t - phi) gives X_n = R cos(phi), Y_n = R sin(phi). The digital lock-in
# realizes the same definition as X_n = LP[2 S cos(n w t)], Y_n = LP[2 S sin(n w t)],
# DC = LP[S].

"""
Supertype of WMS demodulators (`HarmonicExpansion`, `DigitalLockIn`): how the in-phase and
quadrature components X_n, Y_n and the DC level are extracted from the detector signal.
"""
abstract type AbstractDemodulator end

function check_harmonics(harmonics)
    all(n -> n >= 1, harmonics) || throw(ArgumentError("harmonics must be positive integers, got $harmonics"))
    allunique(harmonics) || throw(ArgumentError("harmonics must be unique, got $harmonics"))
    return Tuple(Int.(harmonics))
end

"""
    HarmonicExpansion(; harmonics = (1, 2), n_phase_points = 64)

Quasi-static demodulator: at each slow-scan time (the measurement coordinate, s) the slow
quantities are frozen and the Fourier coefficients `X_n`, `Y_n` and `DC` of the detector signal
over one modulation period are computed by the trapezoid rule on `n_phase_points` equally
spaced reference angles (spectrally accurate for periodic integrands). No filter lag. Requires
`n_phase_points > 2 * maximum(harmonics)`.

```julia
demodulator = HarmonicExpansion(harmonics = (1, 2), n_phase_points = 64)
```
"""
struct HarmonicExpansion{N} <: AbstractDemodulator
    harmonics::NTuple{N,Int}
    n_phase_points::Int
    function HarmonicExpansion(harmonics, n_phase_points::Integer)
        h = check_harmonics(harmonics)
        n_phase_points > 2 * maximum(h) ||
            throw(ArgumentError("HarmonicExpansion: n_phase_points must exceed 2 * maximum(harmonics)"))
        return new{length(h)}(h, Int(n_phase_points))
    end
end
HarmonicExpansion(; harmonics = (1, 2), n_phase_points::Integer = 64) =
    HarmonicExpansion(harmonics, n_phase_points)

"""
    DigitalLockIn(; harmonics = (1, 2), lowpass = Butterworth(order = 5, cutoff_Hz = 100.0),
                  sampling_rate_Hz, pre_roll_s = 0.0)

Time-domain demodulator: the detector signal is simulated at `sampling_rate_Hz` from
`first(times) - pre_roll_s` to the last measurement time (s), mixed with `2 cos(n w t)` and
`2 sin(n w t)`, low-pass filtered from a zero initial state, and linearly interpolated to the
measurement times. Use `pre_roll_s` of several filter time constants to let the filter settle
before the first observation. The sampling rate must resolve the harmonics of the detector
signal that matter: harmonic k aliases onto harmonic n when `k f` folds onto `n f` about a
multiple of `sampling_rate_Hz`, as in real hardware without an anti-aliasing filter. A measured trace on the same sample times
(`lockin_sample_times(prepared)`) goes through the identical filters with `demodulate`.

```julia
demodulator = DigitalLockIn(sampling_rate_Hz = 200e3, pre_roll_s = 0.05)
```
"""
struct DigitalLockIn{N} <: AbstractDemodulator
    harmonics::NTuple{N,Int}
    lowpass::Butterworth
    sampling_rate_Hz::Float64
    pre_roll_s::Float64
    function DigitalLockIn(harmonics, lowpass::Butterworth, sampling_rate_Hz::Real, pre_roll_s::Real)
        h = check_harmonics(harmonics)
        sampling_rate_Hz > 0 || throw(ArgumentError("DigitalLockIn: sampling_rate_Hz must be positive"))
        pre_roll_s >= 0 || throw(ArgumentError("DigitalLockIn: pre_roll_s must be non-negative"))
        return new{length(h)}(h, lowpass, Float64(sampling_rate_Hz), Float64(pre_roll_s))
    end
end
DigitalLockIn(; harmonics = (1, 2), lowpass::Butterworth = Butterworth(order = 5, cutoff_Hz = 100.0),
              sampling_rate_Hz::Real, pre_roll_s::Real = 0.0) =
    DigitalLockIn(harmonics, lowpass, sampling_rate_Hz, pre_roll_s)

"""
    HarmonicSignals

Demodulator output at the measurement times: `harmonics` (tuple of n), `X` and `Y` (one
column per harmonic, in detector units) and `dc`. Read with `in_phase(h, n)`,
`quadrature(h, n)`, `magnitude(h, n)` and `dc_level(h)`.
"""
struct HarmonicSignals{N, T}
    harmonics::NTuple{N,Int}
    X::Matrix{T}
    Y::Matrix{T}
    dc::Vector{T}
end

function harmonic_column(h::HarmonicSignals, n::Integer)
    j = findfirst(==(n), h.harmonics)
    j === nothing && throw(ArgumentError("harmonic $n was not demodulated; available: $(h.harmonics)"))
    return j
end

"""`in_phase(h::HarmonicSignals, n)`: `X_n` at each measurement time."""
in_phase(h::HarmonicSignals, n::Integer) = h.X[:, harmonic_column(h, n)]
"""`quadrature(h::HarmonicSignals, n)`: `Y_n` at each measurement time."""
quadrature(h::HarmonicSignals, n::Integer) = h.Y[:, harmonic_column(h, n)]
"""`magnitude(h::HarmonicSignals, n)`: `sqrt(X_n^2 + Y_n^2)` at each measurement time."""
function magnitude(h::HarmonicSignals, n::Integer)
    j = harmonic_column(h, n)
    return @views @. sqrt(h.X[:, j]^2 + h.Y[:, j]^2)
end
"""`dc_level(h::HarmonicSignals)`: mean (low-passed) detector level at each measurement time."""
dc_level(h::HarmonicSignals) = h.dc

# ---------------------------------------------------------------- the instrument

"""
    WavelengthModulation(tuning::LaserTuning, laser::IntensityModulation, demodulator, observable;
                         detector_gain = 1.0, detector_offset = 0.0)

Wavelength-modulation spectroscopy instrument. The detector signal is
`detector_gain * laser_intensity(t) * transmission(wavenumber_cm(t)) + detector_offset`; the
demodulator (`HarmonicExpansion` or `DigitalLockIn`) extracts harmonics and the observable
(`HarmonicRatio`, `HarmonicOverDC`, `HarmonicProjection`, `Quadratures`,
`BackgroundSubtracted`) is compared with the data. The measurement coordinate is the slow-scan
time in s. `detector_gain` and `detector_offset` are slots (state paths
`(:instrument, :detector_gain)`, `(:instrument, :detector_offset)`).

```julia
instrument = WavelengthModulation(
    LaserTuning(FixedCenter(6046.95), SinusoidalModulation(0.05, 10e3)),
    IntensityModulation(mean = 1.0, linear_amplitude = 0.1),
    HarmonicExpansion(), HarmonicRatio(2, 1))
```
"""
struct WavelengthModulation{T<:LaserTuning, L<:IntensityModulation, D<:AbstractDemodulator, O,
                            G<:AbstractParameter, F<:AbstractParameter} <: AbstractInstrument
    tuning::T
    laser::L
    demodulator::D
    observable::O
    detector_gain::G
    detector_offset::F
    function WavelengthModulation(tuning::T, laser::IntensityModulation, demodulator::D,
                                  observable::O, gain::G, offset::F) where {T, D, O, G, F}
        missing_n = setdiff(required_harmonics(observable), demodulator.harmonics)
        isempty(missing_n) ||
            throw(ArgumentError("WavelengthModulation: observable needs harmonics $missing_n, not in the demodulator's $(demodulator.harmonics)"))
        if demodulator isa DigitalLockIn
            fmax = maximum(demodulator.harmonics) * tuning.modulation.frequency_Hz
            demodulator.sampling_rate_Hz > 2fmax ||
                throw(ArgumentError("WavelengthModulation: sampling_rate_Hz must exceed twice the highest harmonic frequency $fmax Hz"))
        end
        l = attach_modulation(laser, tuning.modulation)
        return new{T,typeof(l),D,O,G,F}(tuning, l, demodulator, observable, gain, offset)
    end
end
WavelengthModulation(tuning, laser, demodulator, observable; detector_gain = 1.0,
                     detector_offset = 0.0) =
    WavelengthModulation(tuning, laser, demodulator, observable, as_parameter(detector_gain),
                         as_parameter(detector_offset))

function bind_parameters(w::WavelengthModulation, x_prior)
    return WavelengthModulation(bind_parameters(w.tuning, x_prior), bind_parameters(w.laser, x_prior),
                                w.demodulator, bind_parameters(w.observable, x_prior),
                                bind_parameter(w.detector_gain, (:instrument, :detector_gain), x_prior),
                                bind_parameter(w.detector_offset, (:instrument, :detector_offset), x_prior))
end

# whether the transmission-1 background depends on the state
background_depends_on_state(w::WavelengthModulation) =
    has_state_slots(w.laser) || has_state_slots(w.detector_gain) ||
    has_state_slots(w.detector_offset) || has_state_slots(w.observable)

# ---------------------------------------------------------------- prepared forms

struct PreparedExpansion
    theta::Vector{Float64}       # reference angles w t over one period, rad
    cos_n::Matrix{Float64}       # (2 / M) cos(n theta_j), M x N
    sin_n::Matrix{Float64}       # (2 / M) sin(n theta_j)
end

struct PreparedLockIn{R<:AbstractRange}
    sample_times_s::R
    cos_ref::Matrix{Float64}     # 2 cos(n w t), samples x N
    sin_ref::Matrix{Float64}     # 2 sin(n w t)
    sos::SecondOrderSections
    index::Vector{Int}           # measurement time k lies between samples index[k], index[k] + 1
    weight::Vector{Float64}      # linear weight of sample index[k] + 1
end

"""
    PreparedWavelengthModulation

`prepare(::WavelengthModulation, measurement)` result: the instrument, the slow-scan times
(s), the prepared demodulator and, for a `BackgroundSubtracted` observable with fixed laser
parameters, the cached transmission-1 background (otherwise `nothing`).
"""
struct PreparedWavelengthModulation{I<:WavelengthModulation, D, B}
    instrument::I
    times_s::Vector{Float64}
    demodulator::D
    background::B
end

prepare_demodulator(d::HarmonicExpansion, tuning, times) =
    let M = d.n_phase_points, theta = [2pi * (j - 1) / M for j in 1:M]
        PreparedExpansion(theta, [2 / M * cos(n * th) for th in theta, n in d.harmonics],
                          [2 / M * sin(n * th) for th in theta, n in d.harmonics])
    end

# Lock-in sample grid (s) for the measurement times, and for each time the sample index i and
# linear weight w of sample i + 1 (the time lies between samples i and i + 1).
function lockin_sampling(d::DigitalLockIn, times)
    fs = d.sampling_rate_Hz
    t0 = first(times) - d.pre_roll_s
    n = floor(Int, (maximum(times) - t0) * fs + 1e-9) + 2
    t = range(t0; step = 1 / fs, length = n)
    index = Vector{Int}(undef, length(times))
    weight = Vector{Float64}(undef, length(times))
    for (k, tk) in enumerate(times)
        tk >= t0 || throw(ArgumentError("DigitalLockIn: measurement times must be increasing from the first"))
        u = (tk - t0) * fs
        i = min(floor(Int, u), n - 2)
        index[k] = i + 1
        weight[k] = u - i
    end
    return t, index, weight
end

function prepare_demodulator(d::DigitalLockIn, tuning, times)
    t, index, weight = lockin_sampling(d, times)
    w = 2pi * tuning.modulation.frequency_Hz
    return PreparedLockIn(t, [2cos(n_h * w * ti) for ti in t, n_h in d.harmonics],
                          [2sin(n_h * w * ti) for ti in t, n_h in d.harmonics],
                          design_filter(d.lowpass, d.sampling_rate_Hz), index, weight)
end

"""
    prepare(instrument::WavelengthModulation, measurement) -> PreparedWavelengthModulation

Build the demodulator tables (phase grid, or lock-in references, filter coefficients and
sampling weights) for the measurement's slow-scan times (`measurement.coordinate`, s). For
`Quadratures` the measurement stacks blocks `X_n1, Y_n1, X_n2, ...`, each on the same times,
and the coordinate repeats the times in every block.
"""
function prepare(w::WavelengthModulation, measurement::Measurement)
    coordinate = collect(Float64, measurement.coordinate)
    m = observation_multiplicity(w.observable)
    length(coordinate) % m == 0 ||
        throw(DimensionMismatch("WavelengthModulation: $(length(coordinate)) observations are not a multiple of $m blocks"))
    K = length(coordinate) ÷ m
    times = coordinate[1:K]
    for b in 2:m
        coordinate[(b-1)*K+1:b*K] == times ||
            throw(ArgumentError("WavelengthModulation: every observation block must repeat the same times"))
    end
    demod = prepare_demodulator(w.demodulator, w.tuning, times)
    partial = PreparedWavelengthModulation(w, times, demod, nothing)
    return with_cached_background(partial)
end

function with_cached_background(p::PreparedWavelengthModulation)
    w = p.instrument
    (w.observable isa BackgroundSubtracted && !background_depends_on_state(w)) || return p
    signals = model_harmonics(w, p.demodulator, p.times_s, nothing, nothing, nothing)
    background = collect(Float64, evaluate_observable(w.observable.observable, signals, nothing))
    return PreparedWavelengthModulation(w, p.times_s, p.demodulator, background)
end

function bind_parameters(p::PreparedWavelengthModulation, x_prior)
    w = bind_parameters(p.instrument, x_prior)
    return with_cached_background(PreparedWavelengthModulation(w, p.times_s, p.demodulator, nothing))
end

n_observations(p::PreparedWavelengthModulation) =
    length(p.times_s) * observation_multiplicity(p.instrument.observable)

query_times(p::PreparedWavelengthModulation{<:Any,PreparedExpansion}) = p.times_s
query_times(p::PreparedWavelengthModulation{<:Any,<:PreparedLockIn}) = p.demodulator.sample_times_s

"""`lockin_sample_times(prepared)`: acquisition times (s) of the simulated or measured trace."""
lockin_sample_times(p::PreparedWavelengthModulation{<:Any,<:PreparedLockIn}) = p.demodulator.sample_times_s

function wavenumber_coverage_cm(p::PreparedWavelengthModulation, x_prior)
    tuning = p.instrument.tuning
    a = abs(resolve(tuning.modulation.amplitude_cm, x_prior))
    nu = [scan_wavenumber_cm(tuning.scan, t, x_prior) for t in query_times(p)]
    return (Float64(minimum(nu) - a), Float64(maximum(nu) + a))
end

# ---------------------------------------------------------------- detector signal and harmonics

transmission_at(::Nothing, ::Nothing, nu) = fill(one(eltype(nu)), size(nu))
transmission_at(grid::AbstractRange, transmission::AbstractVector, nu) =
    reshape(interpolate_fine_grid(grid, transmission, vec(nu)), size(nu))

# quasi-static expansion: S[j, k] at reference angle theta_j and slow-scan time t_k
function model_harmonics(w::WavelengthModulation, d::PreparedExpansion, times, grid, transmission, x)
    tuning, laser = w.tuning, w.laser
    a = resolve(tuning.modulation.amplitude_cm, x)
    theta_m = d.theta .+ resolve(tuning.modulation.phase_rad, x)
    nu_scan = [scan_wavenumber_cm(tuning.scan, t, x) for t in times]
    nu = transpose(nu_scan) .+ a .* cos.(theta_m)
    factor = [intensity_modulation_factor(laser, th, x) for th in theta_m]
    slow = [slow_intensity(laser, t, x) for t in times]
    gain = resolve(w.detector_gain, x)
    S = gain .* (factor .* transpose(slow)) .* transmission_at(grid, transmission, nu) .+
        resolve(w.detector_offset, x)
    X = transpose(S) * d.cos_n
    Y = transpose(S) * d.sin_n
    dc = vec(sum(S; dims = 1)) ./ length(d.theta)
    return HarmonicSignals(w.demodulator.harmonics, Matrix(X), Matrix(Y), dc)
end

function model_harmonics(w::WavelengthModulation, d::PreparedLockIn, times, grid, transmission, x)
    t = d.sample_times_s
    nu = [wavenumber_cm(w.tuning, ti, x) for ti in t]
    intensity = [laser_intensity(w.laser, ti, x) for ti in t]
    S = resolve(w.detector_gain, x) .* intensity .* transmission_at(grid, transmission, nu) .+
        resolve(w.detector_offset, x)
    return lockin_harmonics(d, w.demodulator.harmonics, S)
end

function sample_at_times(d::PreparedLockIn, y::AbstractVector)
    return [(1 - d.weight[k]) * y[d.index[k]] + d.weight[k] * y[d.index[k]+1] for k in eachindex(d.index)]
end

function lockin_harmonics(d::PreparedLockIn, harmonics::NTuple{N,Int}, S::AbstractVector) where {N}
    length(S) == length(d.sample_times_s) ||
        throw(DimensionMismatch("detector trace has $(length(S)) samples, the lock-in expects $(length(d.sample_times_s))"))
    dc = sample_at_times(d, filter_signal(d.sos, S))
    T = eltype(dc)
    X = Matrix{T}(undef, length(d.index), N)
    Y = Matrix{T}(undef, length(d.index), N)
    for j in 1:N
        X[:, j] = sample_at_times(d, filter_signal(d.sos, S .* view(d.cos_ref, :, j)))
        Y[:, j] = sample_at_times(d, filter_signal(d.sos, S .* view(d.sin_ref, :, j)))
    end
    return HarmonicSignals(harmonics, X, Y, dc)
end

"""
    harmonic_signals(prepared::PreparedWavelengthModulation, wavenumber_fine_cm, transmission, x)
        -> HarmonicSignals

Modelled `X_n`, `Y_n` and `DC` at the measurement times for fine-grid `transmission` on the
uniform grid `wavenumber_fine_cm` (cm-1) and state `x`. Generic in element type.
"""
harmonic_signals(p::PreparedWavelengthModulation, wavenumber_fine_cm::AbstractRange,
                 transmission::AbstractVector, x) =
    model_harmonics(p.instrument, p.demodulator, p.times_s, wavenumber_fine_cm, transmission, x)

"""
    demodulate(prepared, detector_signal) -> HarmonicSignals

Demodulate a measured detector trace sampled at `lockin_sample_times(prepared)` through the
same references, filters and sampling as the `DigitalLockIn` model.

```julia
p = prepare(instrument, measurement)          # instrument uses DigitalLockIn
h = demodulate(p, trace)
in_phase(h, 2)
```
"""
demodulate(p::PreparedWavelengthModulation{<:Any,<:PreparedLockIn}, detector_signal::AbstractVector) =
    lockin_harmonics(p.demodulator, p.instrument.demodulator.harmonics, detector_signal)

background_signal(p::PreparedWavelengthModulation{<:Any,<:Any,<:AbstractVector}, x) = p.background
function background_signal(p::PreparedWavelengthModulation{<:Any,<:Any,Nothing}, x)
    w = p.instrument
    signals = model_harmonics(w, p.demodulator, p.times_s, nothing, nothing, x)
    return evaluate_observable(w.observable.observable, signals, x)
end

"""
    apply_instrument(prepared::PreparedWavelengthModulation, wavenumber_fine_cm, transmission, x)

WMS observable at the measurement times: transmission is interpolated (cubic B-spline,
unsorted query points) at the modulated wavenumbers, the detector signal is demodulated and
the observable evaluated. Length `n_observations(prepared)`.
"""
function apply_instrument(p::PreparedWavelengthModulation, wavenumber_fine_cm::AbstractRange,
                          transmission::AbstractVector, x)
    signals = harmonic_signals(p, wavenumber_fine_cm, transmission, x)
    return observable_output(p, p.instrument.observable, signals, x)
end

export AbstractDemodulator, HarmonicExpansion, DigitalLockIn, HarmonicSignals
export in_phase, quadrature, magnitude, dc_level
export WavelengthModulation, PreparedWavelengthModulation, harmonic_signals, demodulate
export lockin_sample_times
