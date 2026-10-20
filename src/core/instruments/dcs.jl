# Dual-comb spectroscopy instrument (section 3.6 of docs/architecture_plan.md): fine-grid
# transmission is interpolated onto the measured wavenumbers with a cubic B-spline. When the
# query wavenumbers do not depend on the state (`NoCalibration`, `LinearStretch` with a fixed
# stretch), `bind_fine_grid` stores the spline as a sparse matrix (`interpolation_matrix`,
# equal to the spline to ~1e-15); a stretch read from the state keeps the per-call spline,
# since the query points then move with the state.

"""
    NoCalibration()

DCS wavenumber calibration that leaves the measured wavenumbers unchanged.

```julia
DCS(calibration = NoCalibration())
```
"""
struct NoCalibration end

"""
    LinearStretch(; stretch = 1.0)

DCS wavenumber calibration: a model wavenumber `nu` (cm-1) is observed at `stretch * nu`, so
the observation at measured wavenumber `nu_obs` is the transmission at `nu_obs / stretch`.
`stretch` (dimensionless) is a parameter slot: a number becomes `Fixed`, and
`bind_parameters` reads it from the state entry `x.instrument.stretch` when present. Replaces
the v0.1 `"shift"` key.

```julia
instrument = DCS(calibration = LinearStretch())
x_prior = ComponentVector(CH4 = 4e19, instrument = (stretch = 1.0,))
```
"""
struct LinearStretch{S<:AbstractParameter}
    stretch::S
end
LinearStretch(; stretch = 1.0) = LinearStretch(as_parameter(stretch))

"""State path of the `LinearStretch` slot."""
const STRETCH_STATE_PATH = (:instrument, :stretch)

"""
    DCS(; calibration = NoCalibration())

Dual-comb spectrometer: the observable is the transmission interpolated (cubic B-spline, as
v0.1) onto the measured wavenumbers (cm-1) of the measurement. `calibration` is
`NoCalibration()` or `LinearStretch(...)`. `prepare(DCS(), measurement)` returns a
`PreparedDCS`. For `NoCalibration` and a fixed stretch, `bind_fine_grid` (called by the
builder) precomputes the spline as a sparse matrix, so `apply_instrument` is one sparse
matrix-vector product; a stretch read from the state, or an unbound `PreparedDCS`, evaluates
the spline on each call.

```julia
prepared = prepare(DCS(), measurement)
grid = default_fine_grid(prepared; oversampling = 2)
prepared = bind_fine_grid(prepared, grid)
signal = apply_instrument(prepared, grid, exp.(-tau), x)
```
"""
struct DCS{C} <: AbstractInstrument
    calibration::C
end
DCS(; calibration = NoCalibration()) = DCS(calibration)

"""
    PreparedDCS

A `DCS` prepared for one measurement: its calibration, observed wavenumbers (cm-1, in the
measurement's order, which need not be sorted), and after `bind_fine_grid` the fine grid and
the sparse interpolation matrix (both `nothing` before, and for a stretch read from the
state).

```julia
prepared = prepare(DCS(), measurement)
n_observations(prepared) == length(measurement.y)
```
"""
struct PreparedDCS{C, V<:AbstractVector{Float64}, G, M}
    calibration::C
    wavenumber_cm::V
    wavenumber_fine_cm::G
    matrix::M
end
PreparedDCS(calibration, wavenumber_cm::AbstractVector{Float64}) =
    PreparedDCS(calibration, wavenumber_cm, nothing, nothing)

function prepare(instrument::DCS, m::Measurement)
    nu = collect(Float64, m.coordinate)
    all(isfinite, nu) || throw(ArgumentError("DCS: measurement wavenumbers must be finite"))
    return PreparedDCS(instrument.calibration, nu)
end
prepare(prepared::PreparedDCS, m::Measurement) = prepare(DCS(prepared.calibration), m)

n_observations(p::PreparedDCS) = length(p.wavenumber_cm)

"""Model wavenumbers (cm-1) at which the transmission is sampled, for state `x`."""
@inline query_wavenumbers(p::PreparedDCS{NoCalibration}, x) = p.wavenumber_cm
@inline query_wavenumbers(p::PreparedDCS{<:LinearStretch}, x) =
    p.wavenumber_cm ./ resolve(p.calibration.stretch, x)

"""
    apply_instrument(prepared::PreparedDCS, wavenumber_fine_cm::AbstractRange, transmission, x)

Transmission on the uniform fine grid `wavenumber_fine_cm` (cm-1) interpolated onto the
observed wavenumbers (divided by the stretch for `LinearStretch`). Generic in the element
types of `transmission` and `x`. The fine grid must cover `wavenumber_coverage_cm`. After
`bind_fine_grid` (no stretch or a fixed one) this is the stored sparse matrix times
`transmission`, and `wavenumber_fine_cm` must be the bound grid; otherwise the cubic
B-spline is evaluated at the query wavenumbers.

```julia
apply_instrument(prepare(DCS(), measurement), grid, exp.(-tau), x)
```
"""
function apply_instrument(p::PreparedDCS, wavenumber_fine_cm::AbstractRange,
                          transmission::AbstractVector, x)
    return interpolate_fine_grid(wavenumber_fine_cm, transmission, query_wavenumbers(p, x))
end
function apply_instrument(p::PreparedDCS{<:Any,<:Any,<:AbstractRange,<:SparseMatrixCSC},
                          wavenumber_fine_cm::AbstractRange, transmission::AbstractVector, x)
    check_bound_grid(p.wavenumber_fine_cm, wavenumber_fine_cm, transmission)
    return p.matrix * transmission
end

"""
    bind_fine_grid(prepared::PreparedDCS, wavenumber_fine_cm::AbstractRange) -> PreparedDCS

Store the fine grid and the sparse cubic-B-spline matrix (`interpolation_matrix`) from it to
the query wavenumbers, for `NoCalibration` and a fixed `LinearStretch`. A stretch read from
the state moves the query points with the state, so `prepared` is returned unchanged and the
spline is evaluated on each call.

```julia
prepared = bind_fine_grid(prepare(DCS(), measurement), grid)
```
"""
bind_fine_grid(p::PreparedDCS, grid::AbstractRange) =
    PreparedDCS(p.calibration, p.wavenumber_cm, grid,
                interpolation_matrix(grid, query_wavenumbers(p, nothing)))
bind_fine_grid(p::PreparedDCS{<:LinearStretch{<:FromState}}, grid::AbstractRange) =
    PreparedDCS(p.calibration, p.wavenumber_cm)

"""
    wavenumber_coverage_cm(prepared::PreparedDCS, x_prior) -> (minimum, maximum)

Model wavenumber range (cm-1) the fine grid must contain: the observed range, divided by the
prior stretch for `LinearStretch`. No margin is added.

```julia
wavenumber_coverage_cm(prepare(DCS(), measurement), x_prior)
```
"""
function wavenumber_coverage_cm(p::PreparedDCS, x_prior)
    nu = query_wavenumbers(p, x_prior)
    return (Float64(minimum(nu)), Float64(maximum(nu)))
end

"""
    default_fine_grid(prepared::PreparedDCS; spacing_cm = nothing, oversampling = 2,
                      margin_cm = 1.0, x_prior = nothing) -> StepRangeLen

Uniform fine grid (cm-1) from the observed range minus/plus `margin_cm`, with spacing
`spacing_cm` or else the median observed spacing divided by `oversampling` (v0.1 used half
the observed spacing and a 1 cm margin). `x_prior` sets the stretch for `LinearStretch`
(defaults to the fixed value, or 1 when the stretch is read from the state).

```julia
grid = default_fine_grid(prepare(DCS(), measurement); spacing_cm = 0.003)
```
"""
function default_fine_grid(p::PreparedDCS; spacing_cm = nothing, oversampling::Real = 2,
                           margin_cm::Real = 1.0, x_prior = nothing)
    margin_cm >= 0 || throw(ArgumentError("margin_cm must be non-negative, got $margin_cm"))
    lo, hi = default_coverage(p, x_prior)
    step = if spacing_cm === nothing
        length(p.wavenumber_cm) >= 2 ||
            throw(ArgumentError("default_fine_grid: give spacing_cm for a single observation"))
        oversampling > 0 || throw(ArgumentError("oversampling must be positive, got $oversampling"))
        median(diff(sort(p.wavenumber_cm))) / oversampling
    else
        Float64(spacing_cm)
    end
    step > 0 || throw(ArgumentError("fine-grid spacing must be positive, got $step"))
    n = ceil(Int, (hi - lo + 2 * margin_cm) / step) + 1
    return range(lo - margin_cm; step = step, length = n)
end

default_coverage(p::PreparedDCS, x_prior) = wavenumber_coverage_cm(p, x_prior)
default_coverage(p::PreparedDCS{NoCalibration}, ::Nothing) = wavenumber_coverage_cm(p, nothing)
default_coverage(p::PreparedDCS{<:LinearStretch{<:Fixed}}, ::Nothing) =
    wavenumber_coverage_cm(p, nothing)
default_coverage(p::PreparedDCS{<:LinearStretch{<:FromState}}, ::Nothing) =
    extrema(p.wavenumber_cm)

bind_parameters(c::NoCalibration, x_prior) = c
bind_parameters(c::LinearStretch, x_prior) =
    LinearStretch(bind_parameter(c.stretch, STRETCH_STATE_PATH, x_prior))
bind_parameters(d::DCS, x_prior) = DCS(bind_parameters(d.calibration, x_prior))
function bind_parameters(p::PreparedDCS, x_prior)
    unbound = PreparedDCS(bind_parameters(p.calibration, x_prior), p.wavenumber_cm)
    # the calibration may change from fixed to state-read: rebuild any bound matrix
    return p.wavenumber_fine_cm === nothing ? unbound : bind_fine_grid(unbound, p.wavenumber_fine_cm)
end

export DCS, PreparedDCS, NoCalibration, LinearStretch
