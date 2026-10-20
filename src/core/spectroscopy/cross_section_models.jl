# Cross-section models backed by vSmartMOM (section 3.4 of docs/architecture_plan.md):
# line-by-line Voigt (`HitranVoigt`) and the precomputed interpolation tables (`LookupTable`).
#
# vSmartMOM's model structs have abstract fields, so its calls are not inferable. Each method
# below converts the conditions to one concrete element type, calls vSmartMOM behind a
# function barrier, and asserts the concrete result type, so `cross_section` itself infers.

# ---------------------------------------------------------------- HitranVoigt

"""
    HitranVoigt(model::vSmartMOM.Absorption.HitranModel; range_cm = line-centre extrema)

Line-by-line cross sections in cm2/molecule from a HITRAN line list, using vSmartMOM's
`absorption_cross_section` with the line shape, wing cutoff (cm-1) and complex error function
stored in `model`. Depends on pressure and temperature. `range_cm` (cm-1) is the interval the
line list was read for; it is what `spectral_range_cm` reports. The wavenumber grid passed to
`cross_section` must be sorted in increasing order.

```julia
import vSmartMOM.Absorption as Absorption
table = Absorption.read_hitran("hit20.par"; mol = 6, iso = 1, ν_min = 6076.0, ν_max = 6084.0)
ch4 = HitranVoigt(Absorption.make_hitran_model(table, Absorption.Voigt()); range_cm = (6076.0, 6084.0))
cross_section(ch4, 6077.0:0.01:6083.0, GasConditions(850.0, 290.0, 0.0))  # cm2/molecule
```
Usually built with `load_cross_section_model(HitranVoigt, path; molecule, isotope, range_cm)`.
"""
struct HitranVoigt <: AbstractCrossSectionModel
    model::Absorption.HitranModel
    range_cm::Tuple{Float64,Float64}
end

function HitranVoigt(model::Absorption.HitranModel;
                     range_cm::Tuple{Real,Real} = line_center_range_cm(model))
    lo, hi = Float64(range_cm[1]), Float64(range_cm[2])
    lo < hi || throw(ArgumentError("HitranVoigt range_cm must be increasing, got ($lo, $hi)"))
    return HitranVoigt(model, (lo, hi))
end

"""Minimum and maximum line-centre wavenumber (cm-1) in a vSmartMOM `HitranModel`."""
function line_center_range_cm(model::Absorption.HitranModel)
    centers = model.hitran.νᵢ
    isempty(centers) && throw(ArgumentError("HitranModel holds no lines"))
    return (Float64(minimum(centers)), Float64(maximum(centers)))
end

depends_on(::HitranVoigt) = (:pressure, :temperature)
spectral_range_cm(m::HitranVoigt) = m.range_cm

"""Common element type of a cross section evaluated at `conditions`: at least Float64."""
condition_eltype(::GasConditions{T}) where {T} = promote_type(T, Float64)

function cross_section(m::HitranVoigt, wavenumber_cm::AbstractVector, c::GasConditions)
    R = condition_eltype(c)
    return hitran_cross_section(m.model, wavenumber_cm, R(c.pressure_hPa),
                                R(c.temperature_K))::Vector{R}
end

# Function barrier around vSmartMOM. With pressure and temperature of one type R, vSmartMOM
# allocates its result as `zeros(R, n)` on the CPU, so the result is a `Vector{R}`.
function hitran_cross_section(model::Absorption.HitranModel, wavenumber_cm,
                              pressure_hPa::R, temperature_K::R) where {R<:Real}
    sigma = Absorption.absorption_cross_section(model, wavenumber_cm, pressure_hPa,
                                                temperature_K)
    return convert(Vector{R}, sigma)
end

# ---------------------------------------------------------------- LookupTable

"""
    LookupTable(model::vSmartMOM.Absorption.InterpolationModel)

Cross sections in cm2/molecule interpolated from a table precomputed over wavenumber (cm-1),
pressure (hPa) and temperature (K), the "Voigt LUT" `.jld2` files made with vSmartMOM's
`make_interpolation_model`. The interpolant is scaled to the model's grids once, at
construction, and stored with a concrete type. Depends on pressure and temperature. Valid
inside the model's wavenumber grid, and for pressure and temperature inside its grids
(evaluation outside them is an error).

```julia
lut = LookupTable(Absorption.make_interpolation_model(table, Absorption.Voigt(),
                  6077.0:0.01:6083.0, 700.0:50.0:1000.0, 260.0:10.0:310.0))
cross_section(lut, 6078.0:0.01:6082.0, GasConditions(850.0, 290.0, 0.0))
```
Usually loaded with `load_cross_section_model(LookupTable, "table.jld2")`.
"""
struct LookupTable{S} <: AbstractCrossSectionModel
    interpolant::S
    model::Absorption.InterpolationModel
    range_cm::Tuple{Float64,Float64}
    pressure_range_hPa::Tuple{Float64,Float64}
    temperature_range_K::Tuple{Float64,Float64}
end

function LookupTable(model::Absorption.InterpolationModel)
    scaled = Interpolations.scale(model.itp, model.ν_grid, model.p_grid, model.t_grid)
    extrema_f64(r) = (Float64(minimum(r)), Float64(maximum(r)))
    return LookupTable(scaled, model, extrema_f64(model.ν_grid), extrema_f64(model.p_grid),
                       extrema_f64(model.t_grid))
end

depends_on(::LookupTable) = (:pressure, :temperature)
spectral_range_cm(m::LookupTable) = m.range_cm

function cross_section(m::LookupTable, wavenumber_cm::AbstractVector, c::GasConditions)
    R = condition_eltype(c)
    p = R(c.pressure_hPa)
    T = R(c.temperature_K)
    check_in_range("pressure_hPa", p, m.pressure_range_hPa)
    check_in_range("temperature_K", T, m.temperature_range_K)
    check_grid_in_range(wavenumber_cm, m.range_cm)
    sigma = Vector{R}(undef, length(wavenumber_cm))
    itp = m.interpolant
    @inbounds for (i, nu) in enumerate(wavenumber_cm)
        sigma[i] = itp(nu, p, T)
    end
    return sigma
end

# ---------------------------------------------------------------- range checks

"""Throw a `DomainError` when `value` lies outside `(lo, hi)`; `name` labels the message."""
function check_in_range(name::AbstractString, value::Real, range::Tuple{Float64,Float64})
    v = strip_duals(value)
    range[1] <= v <= range[2] ||
        throw(DomainError(v, "$name = $v is outside the table range $(range)"))
    return nothing
end

"""Plain value of a possibly nested ForwardDiff dual number."""
strip_duals(x::Real) = x
strip_duals(x::ForwardDiff.Dual) = strip_duals(ForwardDiff.value(x))

"""Throw a `DomainError` when the wavenumber grid (cm-1) leaves `(lo, hi)`."""
function check_grid_in_range(wavenumber_cm::AbstractVector, range::Tuple{Float64,Float64})
    isempty(wavenumber_cm) && return nothing
    lo, hi = extrema(wavenumber_cm)
    (range[1] <= lo && hi <= range[2]) || throw(DomainError((lo, hi),
        "wavenumber grid ($lo, $hi) cm-1 is outside the table range $(range) cm-1"))
    return nothing
end

export HitranVoigt, LookupTable
