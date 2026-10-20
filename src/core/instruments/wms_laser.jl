# WMS laser model (section 3.7 of docs/architecture_plan.md): wavenumber against time (slow
# scan plus sinusoidal modulation) and laser intensity against time. Every physical
# coefficient is a parameter slot (section 3.2); `bind_parameters` binds a slot to the state
# entry `(:instrument, <name>)` named in each docstring.
#
# Phase convention. With modulation frequency f and phase phi, the modulation angle is
# theta_m(t) = 2 pi f t + phi. The wavenumber is scan(t) + a cos(theta_m) and the intensity
# I_slow(t) (1 + i1 cos(theta_m + psi1) + i2 cos(2 theta_m + psi2)), so the intensity phases
# are measured from the wavenumber modulation. Lock-in references are cos(n 2 pi f t).

abstract type AbstractSlowScan end

"""
    FixedCenter(center_cm)
    FixedCenter(; center_cm)

Line-locked slow scan: the unmodulated wavenumber is `center_cm` (cm-1) at all times. Slot:
`center_cm` (state path `(:instrument, :center_cm)`).

```julia
scan = FixedCenter(6046.95)
```
"""
struct FixedCenter{C<:AbstractParameter} <: AbstractSlowScan
    center_cm::C
end
FixedCenter(center_cm) = FixedCenter(as_parameter(center_cm))
FixedCenter(; center_cm) = FixedCenter(center_cm)

"""
    LinearRamp(start_cm, rate_cm_per_s)
    LinearRamp(; start_cm, rate_cm_per_s)

Linear slow scan `start_cm + rate_cm_per_s * t` (cm-1, t in s). Slots: `start_cm`,
`rate_cm_per_s` (state paths `(:instrument, :start_cm)`, `(:instrument, :rate_cm_per_s)`).

```julia
scan = LinearRamp(start_cm = 6046.5, rate_cm_per_s = 2.0)
```
"""
struct LinearRamp{S<:AbstractParameter, R<:AbstractParameter} <: AbstractSlowScan
    start_cm::S
    rate_cm_per_s::R
end
LinearRamp(start_cm, rate_cm_per_s) =
    LinearRamp(as_parameter(start_cm), as_parameter(rate_cm_per_s))
LinearRamp(; start_cm, rate_cm_per_s) = LinearRamp(start_cm, rate_cm_per_s)

"""
    TriangleScan(start_cm, span_cm, period_s)
    TriangleScan(; start_cm, span_cm, period_s)

Symmetric triangle slow scan: `start_cm` at t = 0, rising linearly to `start_cm + span_cm` at
`t = period_s / 2` and back to `start_cm` at `t = period_s`, repeating. Slots: `start_cm`,
`span_cm`, `period_s` (state paths under `:instrument` with the same names).

```julia
scan = TriangleScan(start_cm = 6046.3, span_cm = 0.8, period_s = 0.1)
```
"""
struct TriangleScan{S<:AbstractParameter, W<:AbstractParameter, P<:AbstractParameter} <: AbstractSlowScan
    start_cm::S
    span_cm::W
    period_s::P
end
TriangleScan(start_cm, span_cm, period_s) =
    TriangleScan(as_parameter(start_cm), as_parameter(span_cm), as_parameter(period_s))
TriangleScan(; start_cm, span_cm, period_s) = TriangleScan(start_cm, span_cm, period_s)

"""
    SinusoidalModulation(amplitude_cm, frequency_Hz, phase_rad = 0.0)
    SinusoidalModulation(; amplitude_cm, frequency_Hz, phase_rad = 0.0)

Wavenumber modulation `amplitude_cm * cos(2 pi frequency_Hz t + phase_rad)` (cm-1). Slots:
`amplitude_cm` (state path `(:instrument, :amplitude_cm)`) and `phase_rad` (state path
`(:instrument, :modulation_phase_rad)`, renamed to avoid clashing with a projection phase).
`frequency_Hz` is a plain number because the lock-in references are built from it.

```julia
modulation = SinusoidalModulation(amplitude_cm = 0.05, frequency_Hz = 10e3)
```
"""
struct SinusoidalModulation{A<:AbstractParameter, P<:AbstractParameter}
    amplitude_cm::A
    frequency_Hz::Float64
    phase_rad::P
    function SinusoidalModulation(amplitude_cm::A, frequency_Hz::Real,
                                  phase_rad::P) where {A<:AbstractParameter, P<:AbstractParameter}
        frequency_Hz > 0 ||
            throw(ArgumentError("SinusoidalModulation: frequency_Hz must be positive, got $frequency_Hz"))
        return new{A,P}(amplitude_cm, Float64(frequency_Hz), phase_rad)
    end
end
SinusoidalModulation(amplitude_cm, frequency_Hz::Real, phase_rad = 0.0) =
    SinusoidalModulation(as_parameter(amplitude_cm), frequency_Hz, as_parameter(phase_rad))
SinusoidalModulation(; amplitude_cm, frequency_Hz, phase_rad = 0.0) =
    SinusoidalModulation(amplitude_cm, frequency_Hz, phase_rad)

"""
    LaserTuning(scan, modulation::SinusoidalModulation)

Laser wavenumber against time: a slow scan (`FixedCenter`, `LinearRamp`, `TriangleScan`)
plus a sinusoidal modulation. Evaluate with `wavenumber_cm(tuning, t, x)`.

```julia
tuning = LaserTuning(FixedCenter(6046.95), SinusoidalModulation(0.05, 10e3))
wavenumber_cm(tuning, 0.0, nothing)   # 6047.0
```
"""
struct LaserTuning{S<:AbstractSlowScan, M<:SinusoidalModulation}
    scan::S
    modulation::M
end

"""
    scan_wavenumber_cm(scan, t, x)

Unmodulated (slow-scan) wavenumber in cm-1 at time `t` (s) for state `x`.
"""
scan_wavenumber_cm(s::FixedCenter, t, x) = resolve(s.center_cm, x) + zero(t)
scan_wavenumber_cm(s::LinearRamp, t, x) = resolve(s.start_cm, x) + resolve(s.rate_cm_per_s, x) * t
function scan_wavenumber_cm(s::TriangleScan, t, x)
    u = t / resolve(s.period_s, x)
    phase = u - floor(u)                           # position within the period, in [0, 1)
    return resolve(s.start_cm, x) + resolve(s.span_cm, x) * (1 - abs(2 * phase - 1))
end

"""`modulation_angle_rad(modulation, t, x)`: `2 pi f t + phase_rad`."""
modulation_angle_rad(m::SinusoidalModulation, t, x) = 2pi * m.frequency_Hz * t + resolve(m.phase_rad, x)

"""
    wavenumber_cm(tuning::LaserTuning, t, x)

Instantaneous laser wavenumber in cm-1 at time `t` (s): `scan(t) + a cos(2 pi f t + phi)`.
`x` is the state (any object when every slot is `Fixed`, for example `nothing`).

```julia
tuning = LaserTuning(LinearRamp(6046.0, 1.0), SinusoidalModulation(0.05, 1e3))
wavenumber_cm(tuning, 0.25, nothing)   # 6046.25 + 0.05 cos(2 pi 250)
```
"""
wavenumber_cm(tuning::LaserTuning, t, x) =
    scan_wavenumber_cm(tuning.scan, t, x) +
    resolve(tuning.modulation.amplitude_cm, x) * cos(modulation_angle_rad(tuning.modulation, t, x))

"""
    IntensityModulation(; mean = 1.0, linear_amplitude = 0.0, linear_phase_rad = 0.0,
                        quadratic_amplitude = 0.0, quadratic_phase_rad = 0.0, scan_slope = 0.0)

Laser intensity `mean (1 + scan_slope t) (1 + i1 cos(theta_m + psi1) + i2 cos(2 theta_m + psi2))`
with `i1 = linear_amplitude`, `psi1 = linear_phase_rad`, `i2 = quadratic_amplitude`,
`psi2 = quadratic_phase_rad` (dimensionless and rad), `scan_slope` in 1/s and
`theta_m = 2 pi f t + phi` the wavenumber modulation angle. Every coefficient is a slot whose
state path is `(:instrument, <field name>)`. `WavelengthModulation` attaches the modulation
(frequency and phase) automatically; to evaluate `laser_intensity` on its own, pass
`modulation`.

```julia
laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1, linear_phase_rad = 0.0)
```
"""
struct IntensityModulation{I0, I1, P1, I2, P2, SL, M}
    mean::I0
    linear_amplitude::I1
    linear_phase_rad::P1
    quadratic_amplitude::I2
    quadratic_phase_rad::P2
    scan_slope::SL
    modulation::M          # `nothing` or the SinusoidalModulation of the tuning
end
function IntensityModulation(; mean = 1.0, linear_amplitude = 0.0, linear_phase_rad = 0.0,
                             quadratic_amplitude = 0.0, quadratic_phase_rad = 0.0,
                             scan_slope = 0.0, modulation = nothing)
    return IntensityModulation(as_parameter(mean), as_parameter(linear_amplitude),
                               as_parameter(linear_phase_rad), as_parameter(quadratic_amplitude),
                               as_parameter(quadratic_phase_rad), as_parameter(scan_slope),
                               modulation)
end

"""`attach_modulation(laser, modulation)`: copy of `laser` using `modulation` for its phase."""
attach_modulation(l::IntensityModulation, m::SinusoidalModulation) =
    IntensityModulation(l.mean, l.linear_amplitude, l.linear_phase_rad, l.quadratic_amplitude,
                        l.quadratic_phase_rad, l.scan_slope, m)

"""`slow_intensity(laser, t, x)`: `mean (1 + scan_slope t)`."""
slow_intensity(l::IntensityModulation, t, x) = resolve(l.mean, x) * (1 + resolve(l.scan_slope, x) * t)

"""`intensity_modulation_factor(laser, theta_m, x)`: `1 + i1 cos(theta_m + psi1) + i2 cos(2 theta_m + psi2)`."""
intensity_modulation_factor(l::IntensityModulation, theta_m, x) =
    1 + resolve(l.linear_amplitude, x) * cos(theta_m + resolve(l.linear_phase_rad, x)) +
    resolve(l.quadratic_amplitude, x) * cos(2 * theta_m + resolve(l.quadratic_phase_rad, x))

"""
    laser_intensity(laser::IntensityModulation, t, x)

Laser intensity (units of `mean`) at time `t` (s) for state `x`. Requires the modulation to be
attached (done by `WavelengthModulation`, or pass `modulation` to the constructor).

```julia
mod = SinusoidalModulation(0.05, 1e3)
laser = IntensityModulation(mean = 2.0, linear_amplitude = 0.1, modulation = mod)
laser_intensity(laser, 0.0, nothing)   # 2.0 * 1.1
```
"""
function laser_intensity(l::IntensityModulation, t, x)
    l.modulation === nothing &&
        throw(ArgumentError("laser_intensity: no modulation attached; pass `modulation` to IntensityModulation"))
    theta = modulation_angle_rad(l.modulation, t, x)
    return slow_intensity(l, t, x) * intensity_modulation_factor(l, theta, x)
end

# ---------------------------------------------------------------- binding slots to the state

instrument_path(name::Symbol) = (:instrument, name)

bind_parameters(s::FixedCenter, x_prior) =
    FixedCenter(bind_parameter(s.center_cm, instrument_path(:center_cm), x_prior))
bind_parameters(s::LinearRamp, x_prior) =
    LinearRamp(bind_parameter(s.start_cm, instrument_path(:start_cm), x_prior),
               bind_parameter(s.rate_cm_per_s, instrument_path(:rate_cm_per_s), x_prior))
bind_parameters(s::TriangleScan, x_prior) =
    TriangleScan(bind_parameter(s.start_cm, instrument_path(:start_cm), x_prior),
                 bind_parameter(s.span_cm, instrument_path(:span_cm), x_prior),
                 bind_parameter(s.period_s, instrument_path(:period_s), x_prior))
bind_parameters(m::SinusoidalModulation, x_prior) =
    SinusoidalModulation(bind_parameter(m.amplitude_cm, instrument_path(:amplitude_cm), x_prior),
                         m.frequency_Hz,
                         bind_parameter(m.phase_rad, instrument_path(:modulation_phase_rad), x_prior))
bind_parameters(t::LaserTuning, x_prior) =
    LaserTuning(bind_parameters(t.scan, x_prior), bind_parameters(t.modulation, x_prior))
function bind_parameters(l::IntensityModulation, x_prior)
    m = l.modulation === nothing ? nothing : bind_parameters(l.modulation, x_prior)
    return IntensityModulation(bind_parameter(l.mean, instrument_path(:mean), x_prior),
                               bind_parameter(l.linear_amplitude, instrument_path(:linear_amplitude), x_prior),
                               bind_parameter(l.linear_phase_rad, instrument_path(:linear_phase_rad), x_prior),
                               bind_parameter(l.quadratic_amplitude, instrument_path(:quadratic_amplitude), x_prior),
                               bind_parameter(l.quadratic_phase_rad, instrument_path(:quadratic_phase_rad), x_prior),
                               bind_parameter(l.scan_slope, instrument_path(:scan_slope), x_prior),
                               m)
end

"""`has_state_slots(component)`: whether any parameter slot of `component` reads the state."""
has_state_slots(::Fixed) = false
has_state_slots(::FromState) = true
has_state_slots(::Nothing) = false
has_state_slots(s::FixedCenter) = has_state_slots(s.center_cm)
has_state_slots(s::LinearRamp) = has_state_slots(s.start_cm) || has_state_slots(s.rate_cm_per_s)
has_state_slots(s::TriangleScan) =
    has_state_slots(s.start_cm) || has_state_slots(s.span_cm) || has_state_slots(s.period_s)
has_state_slots(m::SinusoidalModulation) = has_state_slots(m.amplitude_cm) || has_state_slots(m.phase_rad)
has_state_slots(t::LaserTuning) = has_state_slots(t.scan) || has_state_slots(t.modulation)
has_state_slots(l::IntensityModulation) =
    any(has_state_slots, (l.mean, l.linear_amplitude, l.linear_phase_rad, l.quadratic_amplitude,
                          l.quadratic_phase_rad, l.scan_slope, l.modulation))

export FixedCenter, LinearRamp, TriangleScan, SinusoidalModulation, LaserTuning
export IntensityModulation, wavenumber_cm, laser_intensity
