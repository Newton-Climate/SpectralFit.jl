# Build-time work for forward models (section 3.10): prepare the instrument and baseline,
# choose and check the fine grid, bind parameter slots, precompute cross sections, and
# validate the state. Nothing here runs per forward-model call.

"""State entries that are not gases."""
const NON_GAS_STATE_NAMES = (:pressure_hPa, :temperature_K, :instrument, :baseline)

"""State entry that makes each cross-section condition retrieved."""
const CONDITION_STATE_NAMES = (pressure = :pressure_hPa, temperature = :temperature_K, h2o = :H2O)

"""
    gas_state_names(x_prior) -> Tuple{Vararg{Symbol}}

Top-level state names that are gases: everything except `pressure_hPa`, `temperature_K`,
`instrument` and `baseline`.
"""
gas_state_names(x_prior::ComponentVector) =
    Tuple(name for name in propertynames(x_prior) if !(name in NON_GAS_STATE_NAMES))

"""
    is_retrieved(condition::Symbol, x_prior) -> Bool

Whether the cross-section condition `condition` (`:pressure`, `:temperature` or `:h2o`)
changes with the state, so cross sections that depend on it cannot be precomputed.
"""
is_retrieved(condition::Symbol, x_prior) =
    has_state_path(x_prior, getproperty(CONDITION_STATE_NAMES, condition))

"""
    build_forward_model(atmosphere, absorbers, instrument, baseline, space, units,
                        measurement, x_prior; wavenumber_fine_cm = nothing,
                        fine_grid_spacing_cm = nothing) -> ForwardModel

Do all build-time work for one spectral window and return a callable `ForwardModel`:

1. bind parameter slots of the atmosphere and instrument against the names in `x_prior`
   (a slot becomes `FromState` when the state has its entry);
2. prepare the instrument and the baseline for `measurement`;
3. choose the fine grid (cm-1) from the instrument when none is given, check that it covers
   the instrument and lies inside every cross-section model's range, and bind it to the
   instrument;
4. wrap each absorber in `Precomputed` when none of the conditions it depends on are
   retrieved;
5. validate the state: one entry per absorber and no extra gases, one value per layer, and
   baseline coefficients of the right length. Errors name the offending entry.

```julia
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
x_prior = ComponentVector(H2O = 1e21, CH4 = 1e19, baseline = (window_1 = [1.0, 0, 0],))
model = build_forward_model(atm, (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4)),
                            DCS(), LegendreBaseline(3), LinearSpace(), ColumnDensity(),
                            measurement, x_prior; wavenumber_fine_cm = 6077.0:0.01:6083.0)
y_model = model(x_prior)
```
"""
function build_forward_model(atmosphere::AbstractAtmosphere, absorbers::NamedTuple,
                             instrument, baseline::AbstractBaseline,
                             space::AbstractMeasurementSpace, units::AbstractStateUnits,
                             measurement::Measurement, x_prior::ComponentVector;
                             wavenumber_fine_cm::Union{Nothing,AbstractRange} = nothing,
                             fine_grid_spacing_cm::Union{Nothing,Real} = nothing)
    atm = bind_parameters(atmosphere, x_prior)
    prepared = bind_parameters(prepare(bind_parameters(instrument, x_prior), measurement), x_prior)

    grid = wavenumber_fine_cm === nothing ?
        default_fine_grid(prepared; spacing_cm = fine_grid_spacing_cm, x_prior = x_prior) :
        wavenumber_fine_cm
    check_fine_grid(grid, prepared, absorbers, x_prior)
    prepared = bind_fine_grid(prepared, grid)

    validate_state(atm, absorbers, x_prior)
    prepared_baseline = prepare(baseline, measurement)
    validate_baseline(prepared_baseline, x_prior)

    conditions = [layer_conditions(atm, x_prior, layer, units) for layer in 1:n_layers(atm)]
    built = map(a -> maybe_precompute(a, grid, conditions, x_prior), absorbers)
    return ForwardModel(atm, built, prepared, prepared_baseline, space, units, grid)
end

function maybe_precompute(absorber::Absorber, grid, conditions, x_prior)
    model = absorber.model
    any(c -> is_retrieved(c, x_prior), depends_on(model)) && return absorber
    return Absorber(absorber.name, precompute(model, grid, conditions))
end

function check_fine_grid(grid::AbstractRange, prepared, absorbers, x_prior)
    length(grid) >= 4 || throw(ArgumentError("wavenumber_fine_cm: need at least 4 points"))
    step(grid) > 0 || throw(ArgumentError("wavenumber_fine_cm: must be increasing"))
    lo, hi = wavenumber_coverage_cm(prepared, x_prior)
    (first(grid) <= lo && last(grid) >= hi) || throw(ArgumentError(
        "wavenumber_fine_cm $(first(grid))-$(last(grid)) cm-1 does not cover the instrument, " *
        "which needs $lo-$hi cm-1"))
    for (name, absorber) in pairs(absorbers)
        model_lo, model_hi = spectral_range_cm(absorber.model)
        (first(grid) >= model_lo && last(grid) <= model_hi) || throw(ArgumentError(
            "wavenumber_fine_cm $(first(grid))-$(last(grid)) cm-1 is outside the range " *
            "$model_lo-$model_hi cm-1 of absorber $name"))
    end
    return nothing
end

function validate_state(atm::AbstractAtmosphere, absorbers::NamedTuple, x_prior::ComponentVector)
    for (key, absorber) in pairs(absorbers)
        absorber isa Absorber ||
            throw(ArgumentError("absorbers.$key: expected an Absorber, got $(typeof(absorber))"))
        absorber.name == key || throw(ArgumentError(
            "absorbers.$key: key does not match the Absorber name $(absorber.name)"))
    end
    gases = gas_state_names(x_prior)
    for key in keys(absorbers)
        key in gases || throw(ArgumentError("x_prior: no state entry for absorber $key"))
    end
    for gas in gases
        haskey(absorbers, gas) ||
            throw(ArgumentError("x_prior: state entry $gas has no absorber"))
        n = length(getproperty(x_prior, gas))
        n == n_layers(atm) || throw(ArgumentError(
            "x_prior.$gas: has $n values, the atmosphere has $(n_layers(atm)) layers"))
    end
    for (name, path) in ((:pressure_hPa, :pressure_hPa), (:temperature_K, :temperature_K))
        has_state_path(x_prior, path) || continue
        n = length(getproperty(x_prior, path))
        n == n_layers(atm) || throw(ArgumentError(
            "x_prior.$name: has $n values, the atmosphere has $(n_layers(atm)) layers"))
    end
    return nothing
end

validate_baseline(::NoBaseline, x_prior) = nothing
function validate_baseline(baseline::AbstractBaseline, x_prior)
    name = baseline_name(baseline)
    has_state_path(x_prior, (:baseline, name)) ||
        throw(ArgumentError("x_prior: no baseline coefficients at baseline.$name"))
    n = length(x_prior.baseline[name])
    n == n_parameters(baseline) || throw(ArgumentError(
        "x_prior.baseline.$name: has $n coefficients, the baseline has $(n_parameters(baseline))"))
    return nothing
end

export build_forward_model, gas_state_names
