# Typed retrieval configuration (section 3.12). Holds instances of core types, not strings or
# flags; it is read only by the builder, never inside a forward-model call.

"""
    RetrievalConfig(; windows_cm, baseline_coefficients, kwargs...)

How core objects are assembled for one measurement. Fields (units in names):

- `windows_cm::Vector{Tuple{Float64,Float64}}`: spectral windows, closed intervals in cm-1.
- `baseline_coefficients::Vector{Int}`: Legendre coefficients per window (0 for none).
- `baseline_mode`: `:multiplicative` (default) or `:additive`.
- `measurement_space`: `LinearSpace()` (default) or `LogSpace()`.
- `state_units`: `ColumnDensity()` (default) or `VMR()`.
- `instrument`: default `DCS()`.
- `retrieve_pressure`, `retrieve_temperature`: must agree with whether `x_prior` has
  `pressure_hPa` and `temperature_K`; checked by `build_problem`.
- `h2o_broadening::Dict{Symbol,Float64}`: absorber name => coefficient; wraps that absorber in
  `H2OBroadened` at build time.
- `fine_grid_spacing_cm`: fine-grid spacing in cm-1, or `nothing` for the instrument default.
- `masked_windows_cm`, `mask_mode` (`:deweight`, factor `mask_factor`, or `:drop`).
- `baseline_prior`: `:from_measurement` (default; baseline entries of `x_prior` replaced by
  `initial_coefficients`) or `:as_given`.
- `solver`: default `GaussNewton()`.

The atmosphere is a `HomogeneousPath` built from the measurement's `conditions`
(`pressure_hPa`, `temperature_K`, `path_length_cm`). Layered retrievals use
`build_forward_model` directly.

```julia
config = RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [5],
                         fine_grid_spacing_cm = 0.003)
```
"""
struct RetrievalConfig{M<:AbstractMeasurementSpace, U<:AbstractStateUnits,
                       I<:AbstractInstrument, S<:AbstractSolver}
    windows_cm::Vector{Tuple{Float64,Float64}}
    baseline_coefficients::Vector{Int}
    baseline_mode::Symbol
    measurement_space::M
    state_units::U
    instrument::I
    retrieve_pressure::Bool
    retrieve_temperature::Bool
    h2o_broadening::Dict{Symbol,Float64}
    fine_grid_spacing_cm::Union{Nothing,Float64}
    masked_windows_cm::Vector{Tuple{Float64,Float64}}
    mask_mode::Symbol
    mask_factor::Float64
    baseline_prior::Symbol
    solver::S
end

function RetrievalConfig(; windows_cm, baseline_coefficients, baseline_mode::Symbol = :multiplicative,
                         measurement_space::AbstractMeasurementSpace = LinearSpace(),
                         state_units::AbstractStateUnits = ColumnDensity(),
                         instrument::AbstractInstrument = DCS(),
                         retrieve_pressure::Bool = false, retrieve_temperature::Bool = false,
                         h2o_broadening = Dict{Symbol,Float64}(),
                         fine_grid_spacing_cm = nothing,
                         masked_windows_cm = Tuple{Float64,Float64}[],
                         mask_mode::Symbol = :deweight, mask_factor::Real = 1e-5,
                         baseline_prior::Symbol = :from_measurement,
                         solver::AbstractSolver = GaussNewton())
    windows = Tuple{Float64,Float64}[(Float64(a), Float64(b)) for (a, b) in windows_cm]
    isempty(windows) && throw(ArgumentError("windows_cm: need at least one window"))
    for (a, b) in windows
        a < b || throw(ArgumentError("windows_cm: window ($a, $b) must be increasing"))
    end
    coefficients = collect(Int, baseline_coefficients)
    length(coefficients) == length(windows) || throw(ArgumentError(
        "baseline_coefficients: has $(length(coefficients)) entries for $(length(windows)) windows"))
    all(>=(0), coefficients) || throw(ArgumentError("baseline_coefficients: must be non-negative"))
    baseline_mode in (:multiplicative, :additive) ||
        throw(ArgumentError("baseline_mode: must be :multiplicative or :additive, got :$baseline_mode"))
    mask_mode in (:deweight, :drop) ||
        throw(ArgumentError("mask_mode: must be :deweight or :drop, got :$mask_mode"))
    0 < mask_factor <= 1 || throw(ArgumentError("mask_factor: must be in (0, 1], got $mask_factor"))
    baseline_prior in (:from_measurement, :as_given) ||
        throw(ArgumentError("baseline_prior: must be :from_measurement or :as_given, got :$baseline_prior"))
    fine_grid_spacing_cm === nothing || fine_grid_spacing_cm > 0 ||
        throw(ArgumentError("fine_grid_spacing_cm: must be positive"))
    masked = Tuple{Float64,Float64}[(Float64(a), Float64(b)) for (a, b) in masked_windows_cm]
    broadening = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in h2o_broadening)
    spacing = fine_grid_spacing_cm === nothing ? nothing : Float64(fine_grid_spacing_cm)
    return RetrievalConfig(windows, coefficients, baseline_mode, measurement_space, state_units,
                           instrument, retrieve_pressure, retrieve_temperature, broadening,
                           spacing, masked, mask_mode, Float64(mask_factor), baseline_prior,
                           solver)
end

"""`window_name(i)`: baseline state name of window `i`, `:window_i`."""
window_name(i::Integer) = Symbol("window_", i)

function config_baseline(config::RetrievalConfig, i::Integer)
    n = config.baseline_coefficients[i]
    n == 0 && return NoBaseline()
    mode = config.baseline_mode === :multiplicative ? Multiplicative() : Additive()
    return LegendreBaseline(n; mode = mode, name = window_name(i))
end

function config_absorbers(config::RetrievalConfig, absorbers::NamedTuple)
    for name in keys(config.h2o_broadening)
        haskey(absorbers, name) ||
            throw(ArgumentError("h2o_broadening: no absorber named $name"))
    end
    return map(absorbers) do a
        coefficient = get(config.h2o_broadening, a.name, nothing)
        coefficient === nothing ? a : Absorber(a.name, H2OBroadened(a.model, coefficient))
    end
end

function config_atmosphere(measurement::Measurement)
    c = measurement.conditions
    for field in (:pressure_hPa, :temperature_K, :path_length_cm)
        haskey(c, field) || throw(ArgumentError("measurement.conditions: missing $field"))
    end
    return HomogeneousPath(pressure_hPa = c.pressure_hPa, temperature_K = c.temperature_K,
                           path_length_cm = c.path_length_cm)
end

"""Measurement for window `i`, with masked observations dropped when `mask_mode = :drop`."""
function window_measurement(config::RetrievalConfig, measurement::Measurement, i::Integer)
    lo, hi = config.windows_cm[i]
    m = select_window(measurement, lo, hi)
    (config.mask_mode === :drop && !isempty(config.masked_windows_cm)) || return m
    keep = setdiff(eachindex(m.y), find_mask(m.coordinate, config.masked_windows_cm))
    isempty(keep) && throw(ArgumentError("masked_windows_cm: window $i is entirely masked"))
    return Measurement(y = m.y[keep], coordinate = m.coordinate[keep],
                       noise_variance = m.noise_variance[keep], time = m.time,
                       conditions = m.conditions, metadata = m.metadata)
end

"""
    build_forward_model(config::RetrievalConfig, absorbers, measurement, x_prior)

Forward model for every window of `config`: a `ForwardModel` for one window or a
`MultiWindowModel` for several, sharing one state. Window `i` uses baseline state entry
`baseline.window_i`.
"""
function build_forward_model(config::RetrievalConfig, absorbers::NamedTuple,
                             measurement::Measurement, x_prior::ComponentVector)
    atm = config_atmosphere(measurement)
    built_absorbers = config_absorbers(config, absorbers)
    windows = ntuple(length(config.windows_cm)) do i
        build_forward_model(atm, built_absorbers, config.instrument, config_baseline(config, i),
                            config.measurement_space, config.state_units,
                            window_measurement(config, measurement, i), x_prior;
                            fine_grid_spacing_cm = config.fine_grid_spacing_cm)
    end
    return length(windows) == 1 ? windows[1] : MultiWindowModel(windows)
end

"""
    with_initial_baseline(config, measurement, x_prior) -> ComponentVector

`x_prior` with each window's baseline coefficients replaced by `initial_coefficients` from
the measurement (used when `baseline_prior = :from_measurement`).
"""
function with_initial_baseline(config::RetrievalConfig, measurement::Measurement,
                               x_prior::ComponentVector)
    x = copy(x_prior)
    for i in eachindex(config.windows_cm)
        baseline = config_baseline(config, i)
        baseline isa NoBaseline && continue
        name = window_name(i)
        has_state_path(x, (:baseline, name)) ||
            throw(ArgumentError("x_prior: no baseline coefficients at baseline.$name"))
        x.baseline[name] .= initial_coefficients(baseline, config.measurement_space,
                                                 window_measurement(config, measurement, i))
    end
    return x
end

function check_retrieved_conditions(config::RetrievalConfig, x_prior::ComponentVector)
    for (flag, field, name) in ((config.retrieve_pressure, :retrieve_pressure, :pressure_hPa),
                                (config.retrieve_temperature, :retrieve_temperature, :temperature_K))
        flag == has_state_path(x_prior, name) || throw(ArgumentError(
            "$field is $flag but x_prior $(flag ? "has no" : "has a") $name entry"))
    end
    return nothing
end

"""
    build_problem(config, absorbers, measurement, x_prior, prior_sigma) -> RetrievalProblem

One-call set-up for a single measurement: the forward model, the observation vector and its
precision in the configured measurement space (with masks), and the diagonal prior precision
from `prior_sigma` (same axes as `x_prior`). With `baseline_prior = :from_measurement` the
baseline entries of `x_prior` are replaced by `initial_coefficients`, so one prior serves a
whole batch.

```julia
problem = build_problem(config, (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4)),
                        measurement, x_prior, prior_sigma)
result = solve(problem, config.solver)
```
"""
function build_problem(config::RetrievalConfig, absorbers::NamedTuple, measurement::Measurement,
                       x_prior::ComponentVector, prior_sigma::ComponentVector)
    check_retrieved_conditions(config, x_prior)
    prior_precision = prior_precision_from_sigma(prior_sigma, x_prior)
    x0 = config.baseline_prior === :from_measurement ?
        with_initial_baseline(config, measurement, x_prior) : x_prior
    model = build_forward_model(config, absorbers, measurement, x0)
    masked = config.mask_mode === :deweight ? config.masked_windows_cm : Tuple{Float64,Float64}[]
    parts = [window_measurement(config, measurement, i) for i in eachindex(config.windows_cm)]
    y = reduce(vcat, [observation_vector(config.measurement_space, m) for m in parts])
    precision = reduce(vcat, [diag(obs_precision(config.measurement_space, m;
                                                 masked_windows_cm = masked,
                                                 mask_factor = config.mask_factor))
                              for m in parts])
    return RetrievalProblem(model, y, Diagonal(precision), x0, prior_precision)
end

export RetrievalConfig, build_problem
