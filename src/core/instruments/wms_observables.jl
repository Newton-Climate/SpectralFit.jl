# WMS observables (section 3.7 of docs/architecture_plan.md): which normalized harmonic
# quantity of a `HarmonicSignals` is compared with the data. A reference of 0 means the DC
# level; a reference n >= 1 means the magnitude of harmonic n.

"""
Supertype of WMS observables (`HarmonicRatio`, `HarmonicOverDC`, `HarmonicProjection`,
`Quadratures`, `BackgroundSubtracted`): which normalized harmonic quantity is compared with
the data. A new observable implements `evaluate_observable`.
"""
abstract type AbstractWMSObservable end

"""
    HarmonicRatio(n, reference)

`|H_n| / |H_reference|` at each slow-scan time, with `|H_n| = sqrt(X_n^2 + Y_n^2)`; for
example `HarmonicRatio(2, 1)` is 2f/1f, which cancels detector gain and laser power.
`reference = 0` divides by the DC level instead.

```julia
observable = HarmonicRatio(2, 1)
```
"""
struct HarmonicRatio <: AbstractWMSObservable
    n::Int
    reference::Int
    function HarmonicRatio(n::Integer, reference::Integer)
        n >= 1 && reference >= 0 && n != reference ||
            throw(ArgumentError("HarmonicRatio: need n >= 1, reference >= 0 and n != reference"))
        return new(Int(n), Int(reference))
    end
end

"""
    HarmonicOverDC(n)

`|H_n| / DC` at each slow-scan time. Stays finite where the first harmonic passes through zero.

```julia
observable = HarmonicOverDC(2)
```
"""
struct HarmonicOverDC <: AbstractWMSObservable
    n::Int
    function HarmonicOverDC(n::Integer)
        n >= 1 || throw(ArgumentError("HarmonicOverDC: n must be at least 1"))
        return new(Int(n))
    end
end

"""
    HarmonicProjection(n, phase_rad = 0.0, reference = 0)

Signed projection `(X_n cos(phase_rad) + Y_n sin(phase_rad)) / norm`, what a single-output
lock-in reports; `norm` is the DC level for `reference = 0` and `|H_reference|` otherwise.
`phase_rad` (rad) is a slot with state path `(:instrument, :projection_phase_rad)`.

```julia
observable = HarmonicProjection(2, 0.3, 1)   # signed 2f in-phase at 0.3 rad over |1f|
```
"""
struct HarmonicProjection{P<:AbstractParameter} <: AbstractWMSObservable
    n::Int
    phase_rad::P
    reference::Int
    function HarmonicProjection(n::Integer, phase_rad::P, reference::Integer) where {P<:AbstractParameter}
        n >= 1 && reference >= 0 && n != reference ||
            throw(ArgumentError("HarmonicProjection: need n >= 1, reference >= 0 and n != reference"))
        return new{P}(Int(n), phase_rad, Int(reference))
    end
end
HarmonicProjection(n::Integer, phase_rad = 0.0, reference::Integer = 0) =
    HarmonicProjection(n, as_parameter(phase_rad), reference)

"""
    Quadratures(harmonics)

The raw `X_n` and `Y_n` (detector units), stacked as `[X_n1; Y_n1; X_n2; Y_n2; ...]`; the
measurement must repeat the slow-scan times in each block. For diagnosis and tests.

```julia
observable = Quadratures((1, 2))
```
"""
struct Quadratures{N} <: AbstractWMSObservable
    harmonics::NTuple{N,Int}
    Quadratures(harmonics) = (h = check_harmonics(harmonics); new{length(h)}(h))
end

"""
    BackgroundSubtracted(observable)

`observable` minus the same observable computed with transmission 1, which removes the
harmonic content caused by intensity modulation alone. The background is computed once in
`prepare` when the laser, detector and observable slots are all fixed, and at every call
otherwise.

```julia
observable = BackgroundSubtracted(HarmonicRatio(2, 1))
```
"""
struct BackgroundSubtracted{O<:AbstractWMSObservable} <: AbstractWMSObservable
    observable::O
end

# ---------------------------------------------------------------- properties

"""`required_harmonics(observable)`: tuple of harmonics the observable reads (0 is DC, excluded)."""
required_harmonics(o::HarmonicRatio) = o.reference == 0 ? (o.n,) : (o.n, o.reference)
required_harmonics(o::HarmonicOverDC) = (o.n,)
required_harmonics(o::HarmonicProjection) = o.reference == 0 ? (o.n,) : (o.n, o.reference)
required_harmonics(o::Quadratures) = o.harmonics
required_harmonics(o::BackgroundSubtracted) = required_harmonics(o.observable)

"""`observation_multiplicity(observable)`: observations per slow-scan time."""
observation_multiplicity(::AbstractWMSObservable) = 1
observation_multiplicity(o::Quadratures) = 2 * length(o.harmonics)
observation_multiplicity(o::BackgroundSubtracted) = observation_multiplicity(o.observable)

has_state_slots(::AbstractWMSObservable) = false
has_state_slots(o::HarmonicProjection) = has_state_slots(o.phase_rad)
has_state_slots(o::BackgroundSubtracted) = has_state_slots(o.observable)

bind_parameters(o::HarmonicProjection, x_prior) =
    HarmonicProjection(o.n, bind_parameter(o.phase_rad, (:instrument, :projection_phase_rad), x_prior),
                       o.reference)
bind_parameters(o::BackgroundSubtracted, x_prior) = BackgroundSubtracted(bind_parameters(o.observable, x_prior))

# ---------------------------------------------------------------- evaluation

normalizer(h::HarmonicSignals, reference::Int) = reference == 0 ? dc_level(h) : magnitude(h, reference)

"""
    evaluate_observable(observable, signals::HarmonicSignals, x) -> Vector

Observable values at the slow-scan times for demodulated `signals` and state `x`.

```julia
evaluate_observable(HarmonicOverDC(2), signals, x)
```
"""
evaluate_observable(o::HarmonicRatio, h::HarmonicSignals, x) = magnitude(h, o.n) ./ normalizer(h, o.reference)
evaluate_observable(o::HarmonicOverDC, h::HarmonicSignals, x) = magnitude(h, o.n) ./ dc_level(h)
function evaluate_observable(o::HarmonicProjection, h::HarmonicSignals, x)
    phi = resolve(o.phase_rad, x)
    return (in_phase(h, o.n) .* cos(phi) .+ quadrature(h, o.n) .* sin(phi)) ./ normalizer(h, o.reference)
end
function evaluate_observable(o::Quadratures, h::HarmonicSignals, x)
    K = length(h.dc)
    out = similar(h.dc, 2K * length(o.harmonics))
    for (b, n) in enumerate(o.harmonics)
        out[(2b-2)*K+1:(2b-1)*K] = in_phase(h, n)
        out[(2b-1)*K+1:2b*K] = quadrature(h, n)
    end
    return out
end
evaluate_observable(o::BackgroundSubtracted, h::HarmonicSignals, x) =
    throw(ArgumentError("BackgroundSubtracted needs the prepared instrument; use apply_instrument"))

# used by apply_instrument in wms_demodulation.jl
observable_output(p, observable, signals, x) = evaluate_observable(observable, signals, x)
observable_output(p, observable::BackgroundSubtracted, signals, x) =
    evaluate_observable(observable.observable, signals, x) .- background_signal(p, x)

export AbstractWMSObservable, HarmonicRatio, HarmonicOverDC, HarmonicProjection, Quadratures
export BackgroundSubtracted, evaluate_observable
