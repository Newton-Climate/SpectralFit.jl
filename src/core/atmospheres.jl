# Atmospheres and the optical depth along the light path (section 3.5 of
# docs/architecture_plan.md). Pressure in hPa, temperature in K, lengths in cm, columns in
# molecules per cm2.
#
# H2O rule (issue 8, one definition). Column units: the state holds N_H2O; for a homogeneous
# path q = N_H2O / (N_total - N_H2O) with N_total from the ideal gas law, and the dry-air
# column is N_total - N_H2O = N_total / (1 + q). VMR units: q is the state value and the
# dry-air column is N_total / (1 + q). For a layered column the hydrostatic column is already
# dry, so in column units q = N_H2O_layer / N_dry_layer (vertical columns). Without an H2O
# entry in the state the air is dry (q = 0).

# ---------------------------------------------------------------- state helpers

"""Whether the state `x` has a top-level entry `name`; constant-folded for ComponentVectors."""
@inline state_has_entry(x::ComponentArray, name::Symbol) = hasproperty(x, name)
@inline state_has_entry(x, name::Symbol) = false

"""Value of a state entry for one layer: scalars are shared by every layer."""
@inline layer_value(v::Number, layer) = v
@inline layer_value(v::AbstractVector, layer) = v[layer]

@inline absorber_model(a::Absorber) = a.model
@inline absorber_model(m::AbstractCrossSectionModel) = m

# ---------------------------------------------------------------- homogeneous path

"""
    HomogeneousPath(; pressure_hPa, temperature_K, path_length_cm)

One-layer open path of length `path_length_cm` (cm) at uniform pressure (hPa) and
temperature (K). Pressure and temperature are parameter slots: a plain number becomes
`Fixed`, and `bind_parameters` turns it into `FromState{:pressure_hPa}` /
`FromState{:temperature_K}` when the state has that entry. The slant factor is 1 and the
dry-air column comes from the ideal gas law (see the H2O rule at the top of this file).

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
dry_air_column(atm)   # molecules/cm2 of a dry path, about 4.1e24
```
"""
struct HomogeneousPath{P<:AbstractParameter, T<:AbstractParameter} <: AbstractAtmosphere
    pressure_hPa::P
    temperature_K::T
    path_length_cm::Float64
    function HomogeneousPath{P,T}(p::P, t::T, path_length_cm::Real) where {P,T}
        path_length_cm > 0 ||
            throw(ArgumentError("path_length_cm must be positive, got $path_length_cm"))
        check_positive_slot(p, "pressure_hPa")
        check_positive_slot(t, "temperature_K")
        return new{P,T}(p, t, Float64(path_length_cm))
    end
end
HomogeneousPath(p::P, t::T, path_length_cm::Real) where {P<:AbstractParameter,T<:AbstractParameter} =
    HomogeneousPath{P,T}(p, t, path_length_cm)
HomogeneousPath(; pressure_hPa, temperature_K, path_length_cm) =
    HomogeneousPath(as_parameter(pressure_hPa), as_parameter(temperature_K), path_length_cm)

check_positive_slot(p::FromState, name) = nothing
function check_positive_slot(p::Fixed, name)
    all(v -> v > 0, p.value) || throw(ArgumentError("$name must be positive, got $(p.value)"))
    return nothing
end

n_layers(::HomogeneousPath) = 1
slant_factor(::HomogeneousPath, layer) = 1.0

"""Total (wet) air column of the path for state `x`, molecules/cm2."""
total_air_column(atm::HomogeneousPath, x) =
    homogeneous_air_column(resolve(atm.pressure_hPa, x), resolve(atm.temperature_K, x),
                           atm.path_length_cm)

h2o_dry_mole_fraction(::ColumnDensity, atm::HomogeneousPath, x, layer, n_total) =
    state_has_entry(x, :H2O) ?
        (n_h2o = layer_value(x.H2O, layer); n_h2o / (n_total - n_h2o)) : zero(n_total)
h2o_dry_mole_fraction(::VMR, atm::HomogeneousPath, x, layer, n_total) =
    state_has_entry(x, :H2O) ? layer_value(x.H2O, layer) + zero(n_total) : zero(n_total)

# Per-layer quantities, computed together: (conditions, slant dry-air column).
function layer_terms(atm::HomogeneousPath, x, layer, units::AbstractStateUnits, profile)
    p = resolve(atm.pressure_hPa, x)
    t = resolve(atm.temperature_K, x)
    n_total = homogeneous_air_column(p, t, atm.path_length_cm)
    q = h2o_dry_mole_fraction(units, atm, x, layer, n_total)
    return GasConditions(p, t, q), n_total / (1 + q)
end
column_profile(::HomogeneousPath, x) = nothing

function bind_parameters(atm::HomogeneousPath, x_prior)
    return HomogeneousPath(bind_parameter(atm.pressure_hPa, :pressure_hPa, x_prior),
                           bind_parameter(atm.temperature_K, :temperature_K, x_prior),
                           atm.path_length_cm)
end

"""
    dry_air_column(atmosphere::HomogeneousPath)

Dry-air column (molecules/cm2) of a dry homogeneous path with fixed pressure and
temperature, for writing priors. Errors when pressure or temperature are read from the state.

```julia
n_air = dry_air_column(HomogeneousPath(pressure_hPa = 850.0, temperature_K = 285.0,
                                       path_length_cm = 195017.0))
x_prior = ComponentVector(CH4 = 1900e-9 * n_air)
```
"""
function dry_air_column(atm::HomogeneousPath{<:Fixed,<:Fixed})
    return homogeneous_air_column(atm.pressure_hPa.value, atm.temperature_K.value,
                                  atm.path_length_cm)
end
dry_air_column(::HomogeneousPath) =
    throw(ArgumentError("dry_air_column(atmosphere) needs fixed pressure and temperature; use dry_air_column(atmosphere, x, 1, units)"))

# ---------------------------------------------------------------- layered column

"""Marker: layer dry-air columns are recomputed from the state pressure on each call."""
struct HydrostaticFromState end

"""
    LayeredColumn(; pressure_hPa::Vector, temperature_K::Vector, solar_zenith_deg = 0.0,
                  dry_air_column_cm2 = nothing)

Vertical column of layers with centre pressures `pressure_hPa` (hPa) and temperatures
`temperature_K` (K), one value per layer, seen at solar zenith angle `solar_zenith_deg`
(degrees). The air mass (slant factor) is `1 / cosd(solar_zenith_deg)`. Layer dry-air
columns (vertical, molecules/cm2) come from hydrostatic balance on half-pressure levels
(`layer_pressure_thickness_hPa`, `hydrostatic_dry_air_column`; needs two or more layers)
unless given explicitly as `dry_air_column_cm2`. Pressure and temperature are parameter
slots (state entries `:pressure_hPa`, `:temperature_K`, each a full per-layer profile, open
question 6). Gas state entries are vectors with one value per layer.

```julia
atm = LayeredColumn(pressure_hPa = collect(1000.0:-100.0:100.0),
                    temperature_K = fill(260.0, 10), solar_zenith_deg = 30.0)
n_layers(atm)          # 10
slant_factor(atm, 1)   # 1 / cosd(30)
```
"""
struct LayeredColumn{P<:AbstractParameter, T<:AbstractParameter, D} <: AbstractAtmosphere
    pressure_hPa::P
    temperature_K::T
    solar_zenith_deg::Float64
    air_mass::Float64
    vertical_dry_air_column_cm2::D     # Vector{Float64} or HydrostaticFromState()
    explicit_columns::Bool
    n_layers::Int
end

function LayeredColumn(; pressure_hPa, temperature_K, solar_zenith_deg::Real = 0.0,
                       dry_air_column_cm2 = nothing)
    p = as_parameter(pressure_hPa isa AbstractVector ? collect(Float64, pressure_hPa) : pressure_hPa)
    t = as_parameter(temperature_K isa AbstractVector ? collect(Float64, temperature_K) : temperature_K)
    0 <= solar_zenith_deg < 90 ||
        throw(ArgumentError("solar_zenith_deg must be in [0, 90), got $solar_zenith_deg"))
    check_positive_slot(p, "pressure_hPa")
    check_positive_slot(t, "temperature_K")
    n = layer_count(p, t, dry_air_column_cm2)
    columns = if dry_air_column_cm2 === nothing
        p isa Fixed || throw(ArgumentError("LayeredColumn: give pressure_hPa as a vector of numbers"))
        hydrostatic_dry_air_column.(layer_pressure_thickness_hPa(p.value))
    else
        length(dry_air_column_cm2) == n ||
            throw(DimensionMismatch("dry_air_column_cm2 has length $(length(dry_air_column_cm2)), expected $n layers"))
        all(v -> v > 0, dry_air_column_cm2) || throw(ArgumentError("dry_air_column_cm2 must be positive"))
        collect(Float64, dry_air_column_cm2)
    end
    return LayeredColumn(p, t, Float64(solar_zenith_deg), 1 / cosd(solar_zenith_deg), columns,
                         dry_air_column_cm2 !== nothing, n)
end

function layer_count(p, t, columns)
    lengths = Int[]
    p isa Fixed && push!(lengths, length(p.value))
    t isa Fixed && push!(lengths, length(t.value))
    columns === nothing || push!(lengths, length(columns))
    isempty(lengths) && throw(ArgumentError("LayeredColumn: the number of layers is unknown; give a pressure or temperature profile"))
    all(==(first(lengths)), lengths) ||
        throw(DimensionMismatch("LayeredColumn: pressure, temperature and column lengths differ: $lengths"))
    return first(lengths)
end

n_layers(atm::LayeredColumn) = atm.n_layers
slant_factor(atm::LayeredColumn, layer) = atm.air_mass

"""Vertical layer dry-air columns (molecules/cm2) for state `x`."""
column_profile(atm::LayeredColumn{P,T,Vector{Float64}}, x) where {P,T} =
    atm.vertical_dry_air_column_cm2
column_profile(atm::LayeredColumn{P,T,HydrostaticFromState}, x) where {P,T} =
    hydrostatic_dry_air_column.(layer_pressure_thickness_hPa(resolve(atm.pressure_hPa, x)))

h2o_dry_mole_fraction(::ColumnDensity, atm::LayeredColumn, x, layer, n_dry_vertical) =
    state_has_entry(x, :H2O) ? layer_value(x.H2O, layer) / n_dry_vertical : zero(n_dry_vertical)
h2o_dry_mole_fraction(::VMR, atm::LayeredColumn, x, layer, n_dry_vertical) =
    state_has_entry(x, :H2O) ? layer_value(x.H2O, layer) + zero(n_dry_vertical) : zero(n_dry_vertical)

function layer_terms(atm::LayeredColumn, x, layer, units::AbstractStateUnits, profile)
    p = layer_value(resolve(atm.pressure_hPa, x), layer)
    t = layer_value(resolve(atm.temperature_K, x), layer)
    n_dry = profile[layer]
    q = h2o_dry_mole_fraction(units, atm, x, layer, n_dry)
    return GasConditions(p, t, q), atm.air_mass * n_dry
end

function bind_parameters(atm::LayeredColumn, x_prior)
    p = bind_parameter(atm.pressure_hPa, :pressure_hPa, x_prior)
    t = bind_parameter(atm.temperature_K, :temperature_K, x_prior)
    columns = (p isa FromState && !atm.explicit_columns) ? HydrostaticFromState() :
              atm.vertical_dry_air_column_cm2
    return LayeredColumn(p, t, atm.solar_zenith_deg, atm.air_mass, columns,
                         atm.explicit_columns, atm.n_layers)
end

# ---------------------------------------------------------------- generic interface

"""
    layer_conditions(atmosphere, x, layer, units) -> GasConditions

Pressure (hPa) and temperature (K) of `layer` resolved through the parameter slots, and the
H2O dry mole fraction (mol/mol) from state `x` in `units` (`ColumnDensity()` or `VMR()`).

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
layer_conditions(atm, ComponentVector(H2O = 0.01, CH4 = 1.9e-6), 1, VMR())
```
"""
layer_conditions(atm::AbstractAtmosphere, x, layer, units::AbstractStateUnits) =
    first(layer_terms(atm, x, layer, units, column_profile(atm, x)))

"""
    dry_air_column(atmosphere, x, layer, units)

Slant dry-air column of `layer` in molecules/cm2 for state `x` in `units`.

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
dry_air_column(atm, ComponentVector(CH4 = 1.9e-6), 1, VMR()) == dry_air_column(atm)
```
"""
dry_air_column(atm::AbstractAtmosphere, x, layer, units::AbstractStateUnits) =
    last(layer_terms(atm, x, layer, units, column_profile(atm, x)))

"""
    gas_column(units, atmosphere, x, gas::Symbol, layer)

Slant column of `gas` in `layer`, molecules/cm2. `ColumnDensity`: the slant factor times the
state value; `VMR`: the state value times the slant dry-air column.

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
gas_column(VMR(), atm, ComponentVector(CH4 = 1.9e-6), :CH4, 1)   # 1.9e-6 * dry_air_column(atm)
```
"""
@inline gas_column(units::AbstractStateUnits, atm::AbstractAtmosphere, x, gas::Symbol, layer) =
    gas_column(units, atm, x, Val(gas), layer)
@inline gas_column(::ColumnDensity, atm::AbstractAtmosphere, x, ::Val{gas}, layer) where {gas} =
    slant_factor(atm, layer) * layer_value(getproperty(x, gas), layer)
@inline gas_column(units::VMR, atm::AbstractAtmosphere, x, ::Val{gas}, layer) where {gas} =
    layer_value(getproperty(x, gas), layer) * dry_air_column(atm, x, layer, units)

# Inner forms with the slant dry-air column already computed.
@inline slant_gas_column(::ColumnDensity, atm, x, ::Val{gas}, layer, dry) where {gas} =
    slant_factor(atm, layer) * layer_value(getproperty(x, gas), layer)
@inline slant_gas_column(::VMR, atm, x, ::Val{gas}, layer, dry) where {gas} =
    layer_value(getproperty(x, gas), layer) * dry

"""
    optical_depth(atmosphere, absorbers::NamedTuple, units, x, wavenumber_cm) -> Vector

Total optical depth on `wavenumber_cm` (cm-1): the sum over layers and absorbers of the slant
gas column (molecules/cm2) times the layer cross section (cm2/molecule). The keys of
`absorbers` are the state names of the gases; values are `Absorber`s (or bare cross-section
models). The loop over absorbers is unrolled; with `Precomputed` absorbers each term is a
scaled column of the stored matrix. Generic in the element type of `x`.

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 1e5)
tau = optical_depth(atm, (CH4 = Absorber(:CH4, ch4_model),), ColumnDensity(),
                    ComponentVector(CH4 = 4e19), 6076.0:0.001:6078.0)
transmission = exp.(-tau)
```
"""
function optical_depth(atm::AbstractAtmosphere, absorbers::NamedTuple, units::AbstractStateUnits,
                       x, wavenumber_cm::AbstractVector)
    profile = column_profile(atm, x)
    conditions, dry = layer_terms(atm, x, 1, units, profile)
    FT = promote_type(eltype(x), typeof(conditions.pressure_hPa), typeof(dry), Float64)
    tau = zeros(FT, length(wavenumber_cm))
    add_layer_optical_depth!(tau, absorbers, units, atm, x, wavenumber_cm, conditions, dry, 1)
    for layer in 2:n_layers(atm)
        conditions, dry = layer_terms(atm, x, layer, units, profile)
        add_layer_optical_depth!(tau, absorbers, units, atm, x, wavenumber_cm, conditions, dry, layer)
    end
    return tau
end

@inline add_layer_optical_depth!(tau, ::NamedTuple{(),Tuple{}}, units, atm, x, nu, c, dry, layer) = tau
@inline function add_layer_optical_depth!(tau, absorbers::NamedTuple{names}, units, atm, x, nu,
                                          conditions, dry, layer) where {names}
    column = slant_gas_column(units, atm, x, Val(first(names)), layer, dry)
    sigma = layer_cross_section(absorber_model(first(absorbers)), nu, conditions, layer)
    tau .+= column .* sigma
    return add_layer_optical_depth!(tau, Base.tail(absorbers), units, atm, x, nu, conditions,
                                    dry, layer)
end

export HomogeneousPath, LayeredColumn
