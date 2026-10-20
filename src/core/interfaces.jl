# Abstract types and generic functions shared across core. Concrete types live in their own
# files; this file only fixes the interfaces in section 3 of docs/architecture_plan.md so the
# pieces can be written and tested independently.

# ---------------------------------------------------------------- cross sections (3.4)

"""
Supertype of cross-section sources. A subtype implements `cross_section`, `depends_on` and
`spectral_range_cm`.
"""
abstract type AbstractCrossSectionModel end

"""
    GasConditions(pressure_hPa, temperature_K, h2o_dry_mole_fraction)

Conditions at which a cross section is evaluated. Fields share one `Real` type, so ForwardDiff
dual numbers pass through.
"""
struct GasConditions{T<:Real}
    pressure_hPa::T
    temperature_K::T
    h2o_dry_mole_fraction::T
end
GasConditions(p::Real, T::Real, q::Real) = GasConditions(promote(p, T, q)...)
GasConditions(; pressure_hPa, temperature_K, h2o_dry_mole_fraction = 0.0) =
    GasConditions(pressure_hPa, temperature_K, h2o_dry_mole_fraction)

"""
    cross_section(model, wavenumber_cm::AbstractVector, conditions::GasConditions) -> AbstractVector

Absorption cross section in cm2 per molecule on `wavenumber_cm`. Must accept any `Real`
element type in `conditions`.
"""
function cross_section end

"""
    depends_on(model) -> Tuple

Subset of `(:pressure, :temperature, :h2o)` naming the conditions the cross section depends on.
"""
function depends_on end

"""
    spectral_range_cm(model) -> (minimum, maximum)

Wavenumber range in cm-1 over which the model is valid.
"""
function spectral_range_cm end

"""
    layer_cross_section(model, wavenumber_cm, conditions, layer)

Cross section for one atmospheric layer. Defaults to `cross_section(model, wavenumber_cm,
conditions)`; `Precomputed` returns its stored column for `layer`.
"""
layer_cross_section(model::AbstractCrossSectionModel, wavenumber_cm, conditions, layer) =
    cross_section(model, wavenumber_cm, conditions)

"""
    Absorber(name::Symbol, model::AbstractCrossSectionModel)

Pairs a state-vector name (`:CH4`) with a cross-section model. Forward models hold absorbers
as a `NamedTuple`, for example `(H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))`.
"""
struct Absorber{M<:AbstractCrossSectionModel}
    name::Symbol
    model::M
end

# ---------------------------------------------------------------- atmospheres (3.5)

"""
Supertype of atmospheres. A subtype implements `n_layers`, `layer_conditions`,
`dry_air_column` and `slant_factor`.
"""
abstract type AbstractAtmosphere end

"""`n_layers(atmosphere)::Int`: number of layers."""
function n_layers end
"""`layer_conditions(atmosphere, x, layer, units)::GasConditions` for state `x`. `units` is
needed because the H2O dry mole fraction is computed from the state differently per unit."""
function layer_conditions end
"""`dry_air_column(atmosphere, x, layer, units)`: slant dry-air column in molecules/cm2."""
function dry_air_column end
"""`slant_factor(atmosphere, layer)`: state column to slant column along the light path."""
function slant_factor end
"""`gas_column(units, atmosphere, x, gas::Symbol, layer)`: slant column of `gas` in molecules/cm2."""
function gas_column end
"""`optical_depth(atmosphere, absorbers, units, x, wavenumber_cm)`: total optical depth on the grid."""
function optical_depth end

"""Supertype of state units: how gas amounts are stored in the state vector."""
abstract type AbstractStateUnits end
"""State gas entries are columns in molecules per cm2."""
struct ColumnDensity <: AbstractStateUnits end
"""State gas entries are dry-air mole fractions in mol/mol."""
struct VMR <: AbstractStateUnits end

# ---------------------------------------------------------------- instruments (3.6)

"""
Supertype of instruments. A subtype implements `prepare`, and its prepared form implements
`apply_instrument`, `wavenumber_coverage_cm` and `n_observations`.
"""
abstract type AbstractInstrument end

"""
    prepare(component, measurement)

Build-time preparation of an instrument or baseline for a measurement's geometry (observed
wavenumbers, line-shape matrices, lock-in references, Legendre matrices). Called once.
"""
function prepare end

"""
    apply_instrument(prepared, wavenumber_fine_cm, transmission, x) -> AbstractVector

Instrument observable, one value per observation, from fine-grid transmission and state `x`.
Generic in element type.
"""
function apply_instrument end

"""`wavenumber_coverage_cm(prepared, x_prior) -> (minimum, maximum)` wavenumber needed, cm-1."""
function wavenumber_coverage_cm end
"""`n_observations(prepared)::Int`."""
function n_observations end
"""`default_fine_grid(prepared; oversampling)`: optional suggested fine wavenumber grid."""
function default_fine_grid end

"""
    bind_fine_grid(prepared, wavenumber_fine_cm::AbstractRange)

Second build-time step for instruments whose operator depends on the fine grid (FTS and
grating line-shape matrices). The builder calls `prepare(instrument, measurement)`, chooses
the fine grid from `wavenumber_coverage_cm` or `default_fine_grid`, then calls this. The
fallback returns `prepared` unchanged.
"""
bind_fine_grid(prepared, wavenumber_fine_cm) = prepared

# ---------------------------------------------------------------- measurement space and baselines (3.8)

"""Supertype of measurement spaces: identity (`LinearSpace`) or natural log (`LogSpace`)."""
abstract type AbstractMeasurementSpace end
"""Measurement space equal to instrument units."""
struct LinearSpace <: AbstractMeasurementSpace end
"""Measurement space equal to the natural log of instrument units."""
struct LogSpace <: AbstractMeasurementSpace end

"""`to_measurement_space(space, instrument_output)`: identity or `log`."""
function to_measurement_space end
"""`observation_vector(space, measurement)`: the stored measurement in the given space."""
function observation_vector end
"""`obs_precision(space, measurement; mask)`: inverse observation covariance."""
function obs_precision end

"""Supertype of baselines."""
abstract type AbstractBaseline end
"""No baseline: the model output is used as is."""
struct NoBaseline <: AbstractBaseline end
"""Baseline mode: the polynomial multiplies the signal (adds in log space)."""
struct Multiplicative end
"""Baseline mode: the polynomial adds to the signal."""
struct Additive end

"""`n_parameters(baseline)::Int`: number of state entries the baseline uses."""
function n_parameters end
"""`apply_baseline(prepared_baseline, space, y, x)`: model in measurement space with baseline."""
function apply_baseline end
"""`initial_coefficients(baseline, space, measurement)`: starting baseline coefficients."""
function initial_coefficients end

# ---------------------------------------------------------------- estimation (3.11)

"""Supertype of solvers (`GaussNewton`, `LevenbergMarquardt`)."""
abstract type AbstractSolver end
"""`solve(problem, solver) -> RetrievalResult`."""
function solve end
"""`report(result)`: plain-text optimal-estimation report with a CSV table."""
function report end

export AbstractCrossSectionModel, GasConditions, cross_section, depends_on, spectral_range_cm
export layer_cross_section, Absorber
export AbstractAtmosphere, n_layers, layer_conditions, dry_air_column, slant_factor
export gas_column, optical_depth
export AbstractStateUnits, ColumnDensity, VMR
export AbstractInstrument, prepare, apply_instrument, wavenumber_coverage_cm, n_observations
export default_fine_grid, bind_fine_grid
export AbstractMeasurementSpace, LinearSpace, LogSpace
export to_measurement_space, observation_vector, obs_precision
export AbstractBaseline, NoBaseline, Multiplicative, Additive
export n_parameters, apply_baseline, initial_coefficients
export AbstractSolver, solve, report
