# Batch retrieval configuration (section 3.13, phase 9). Holds file locations, the reader,
# the noise model, the averaging period, the retrieval configuration, the prior and the
# parallelism; validated at construction with errors that name the field.

"""
    BatchRetrievalConfig(; experiment, data_files = nothing, data_directory = nothing,
                         file_pattern = r"\\.h5\$", reader = DCSReaderConfig(), noise = nothing,
                         averaging_period, conditions = nothing, absorbers, retrieval, x_prior,
                         prior_sigma, output_root = "runs", parallel = :processes, n_workers = 2)

Settings for `BatchRetrieval.run`. Fields:

- `experiment::String`: short name used in the run folder name (letters, digits, `-`, `_`,
  `.`).
- `data_files`: vector of input paths, or `data_directory`: a directory whose files are
  listed (not recursively, sorted by name). Give exactly one of the two.
- `file_pattern::Regex`: only files whose base name matches are processed; others are logged
  and counted as skipped. Applied to `data_files` too.
- `reader::DCSReaderConfig`: how DCS HDF5 files are read (path length, pressure scale, field
  layout).
- `noise`: an `AbstractNoiseModel` applied to the full-spectrum records of each file before
  time averaging (it replaces `reader.noise_model`), or `nothing` to keep
  `reader.noise_model`.
- `averaging_period::Dates.Period`: positive length of the time bins (`time_average`).
- `conditions`: `nothing`, or a `ConditionsTable` whose fields (for example pressure and
  temperature from a tower) replace the recorded conditions of each averaged measurement
  (`apply_conditions`), so a retrieval with fixed p and T uses them.
- `absorbers::NamedTuple`: `Absorber`s, as for `build_problem`.
- `retrieval::RetrievalConfig`: windows, baseline, instrument and solver
  (`retrieval.solver` is used).
- `x_prior`, `prior_sigma`: `ComponentVector`s with the same axes (state units); one prior
  serves the whole batch (baseline entries come from each measurement when
  `retrieval.baseline_prior = :from_measurement`).
- `output_root::String`: directory in which the run folder is created.
- `parallel::Symbol`: `:processes` (default; `pmap` over `n_workers`
  worker processes) or `:serial`.
- `n_workers::Int`: number of worker processes for `:processes` (default 2). Existing workers
  are used first; `run` adds the missing ones (same project, `using SpectralFits`) and removes
  the ones it added when it finishes.

# Example
```julia
config = BatchRetrievalConfig(
    experiment = "merlin-r6",
    data_directory = "../data", file_pattern = r"^2016.*\\.h5\$",
    reader = DCSReaderConfig(path_length_cm = 195017.0),
    noise = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60)]),
    averaging_period = Minute(15),
    absorbers = absorbers, retrieval = retrieval,
    x_prior = x_prior, prior_sigma = prior_sigma,
    output_root = "../data/runs")
```
"""
struct BatchRetrievalConfig{A<:NamedTuple, R<:RetrievalConfig, X<:ComponentVector,
                            N<:Union{Nothing,AbstractNoiseModel}}
    experiment::String
    data_files::Union{Nothing,Vector{String}}
    data_directory::Union{Nothing,String}
    file_pattern::Regex
    reader::DCSReaderConfig
    noise::N
    averaging_period::Dates.Period
    conditions::Union{Nothing,ConditionsTable}
    absorbers::A
    retrieval::R
    x_prior::X
    prior_sigma::X
    output_root::String
    parallel::Symbol
    n_workers::Int
end

function BatchRetrievalConfig(; experiment::AbstractString, data_files = nothing,
                              data_directory = nothing, file_pattern::Regex = r"\.h5$",
                              reader::DCSReaderConfig = DCSReaderConfig(), noise = nothing,
                              averaging_period::Dates.Period, conditions = nothing,
                              absorbers::NamedTuple,
                              retrieval::RetrievalConfig, x_prior::ComponentVector,
                              prior_sigma::ComponentVector, output_root::AbstractString = "runs",
                              parallel::Symbol = :processes, n_workers::Integer = 2)
    _check_experiment(experiment)
    files, directory = _check_inputs(data_files, data_directory)
    noise === nothing || noise isa AbstractNoiseModel ||
        throw(ArgumentError("noise: must be an AbstractNoiseModel or nothing, got $(typeof(noise))"))
    Dates.value(averaging_period) > 0 ||
        throw(ArgumentError("averaging_period: must be positive, got $averaging_period"))
    conditions === nothing || conditions isa ConditionsTable || throw(ArgumentError(
        "conditions: must be a ConditionsTable or nothing, got $(typeof(conditions))"))
    isempty(absorbers) && throw(ArgumentError("absorbers: need at least one absorber"))
    _check_windows(retrieval)
    _check_absorber_coverage(absorbers, retrieval)
    x_prior_f, sigma_f = _check_prior(retrieval, x_prior, prior_sigma)
    isempty(output_root) && throw(ArgumentError("output_root: must not be empty"))
    parallel in (:serial, :processes) ||
        throw(ArgumentError("parallel: must be :serial or :processes, got :$parallel"))
    n_workers >= 1 || throw(ArgumentError("n_workers: must be at least 1, got $n_workers"))
    return BatchRetrievalConfig(String(experiment), files, directory, file_pattern, reader, noise,
                                averaging_period, conditions, absorbers, retrieval, x_prior_f, sigma_f,
                                String(output_root), parallel, Int(n_workers))
end

function _check_experiment(experiment::AbstractString)
    isempty(experiment) && throw(ArgumentError("experiment: must not be empty"))
    occursin(r"^[A-Za-z0-9._-]+$", experiment) || throw(ArgumentError(
        "experiment: \"$experiment\" may contain only letters, digits, '.', '_' and '-'"))
    return nothing
end

function _check_inputs(data_files, data_directory)
    (data_files === nothing) == (data_directory === nothing) &&
        throw(ArgumentError("data_files, data_directory: give exactly one of the two"))
    if data_directory !== nothing
        isdir(data_directory) ||
            throw(ArgumentError("data_directory: \"$data_directory\" is not an existing directory"))
        return nothing, String(data_directory)
    end
    files = String[String(f) for f in data_files]
    isempty(files) && throw(ArgumentError("data_files: the list is empty"))
    for f in files
        isfile(f) || throw(ArgumentError("data_files: \"$f\" does not exist"))
    end
    return files, nothing
end

function _check_windows(retrieval::RetrievalConfig)
    isempty(retrieval.windows_cm) &&
        throw(ArgumentError("retrieval.windows_cm: need at least one window"))
    for (a, b) in retrieval.windows_cm
        a < b || throw(ArgumentError("retrieval.windows_cm: window ($a, $b) is empty"))
    end
    return nothing
end

# The DCS fine grid runs from the first observation minus the margin to the last plus the
# margin (1 cm-1 by default); every absorber must cover it. Checked against the window edges
# with a 0.01 cm-1 allowance, since the observations lie inside the window.
function _check_absorber_coverage(absorbers::NamedTuple, retrieval::RetrievalConfig)
    retrieval.instrument isa DCS || return nothing
    margin = something(retrieval.fine_grid_margin_cm, 1.0)
    for (lo, hi) in retrieval.windows_cm, (name, a) in pairs(absorbers)
        r_lo, r_hi = spectral_range_cm(a.model)
        (r_lo <= lo - margin + 0.01 && r_hi >= hi + margin - 0.01) || throw(ArgumentError(
            "absorbers.$name: covers $r_lo-$r_hi cm-1, but window ($lo, $hi) with the fine-grid " *
            "margin of $margin cm-1 needs about $(lo - margin)-$(hi + margin) cm-1; set " *
            "retrieval.fine_grid_margin_cm or use a wider table"))
    end
    return nothing
end

function _check_prior(retrieval::RetrievalConfig, x_prior::ComponentVector,
                      prior_sigma::ComponentVector)
    getaxes(prior_sigma) == getaxes(x_prior) || throw(ArgumentError(
        "prior_sigma: labels $(state_labels(prior_sigma)) do not match x_prior labels " *
        "$(state_labels(x_prior))"))
    all(isfinite, x_prior) || throw(ArgumentError("x_prior: contains non-finite values"))
    all(s -> isfinite(s) && s > 0, prior_sigma) ||
        throw(ArgumentError("prior_sigma: every standard deviation must be positive and finite"))
    for (i, n) in enumerate(retrieval.baseline_coefficients)
        n == 0 && continue
        path = (:baseline, Symbol("window_", i))
        CoreLayer.has_state_path(x_prior, path) || throw(ArgumentError(
            "x_prior: retrieval.baseline_coefficients needs $n coefficients at baseline.window_$i"))
        m = length(x_prior.baseline[path[2]])
        m == n || throw(ArgumentError(
            "x_prior: baseline.window_$i has $m entries, retrieval.baseline_coefficients[$i] = $n"))
    end
    CoreLayer.check_retrieved_conditions(retrieval, x_prior)
    # plain Vector{Float64} data: `collect` on a ComponentVector returns a ComponentVector,
    # which would nest inside the new one
    return (ComponentVector(Vector{Float64}(getdata(x_prior)), getaxes(x_prior)),
            ComponentVector(Vector{Float64}(getdata(prior_sigma)), getaxes(x_prior)))
end

"""
    input_files(config::BatchRetrievalConfig) -> (selected = Vector{String}, skipped = Vector{String})

The input paths to process (base name matches `config.file_pattern`) and the paths skipped,
in sorted order for `data_directory` and in the given order for `data_files`.

# Example
```julia
files = input_files(config)
files.selected
```
"""
function input_files(config::BatchRetrievalConfig)
    candidates = config.data_files !== nothing ? config.data_files :
                 filter(isfile, [joinpath(config.data_directory, f)
                                 for f in sort(readdir(config.data_directory))])
    matches = [occursin(config.file_pattern, basename(f)) for f in candidates]
    return (selected = candidates[matches], skipped = candidates[.!matches])
end

"""
    configuration_text(config::BatchRetrievalConfig) -> String

Human-readable dump of the configuration, written as `configuration.txt` in the run folder.
Absorbers are listed by name and model type (line lists are not dumped).

# Example
```julia
text = configuration_text(config)
```
"""
function configuration_text(config::BatchRetrievalConfig)
    io = IOBuffer()
    println(io, "# SpectralFits batch retrieval configuration")
    for name in fieldnames(BatchRetrievalConfig)
        value = getfield(config, name)
        if name === :absorbers
            println(io, "absorbers:")
            for (k, a) in pairs(value)
                println(io, "  ", k, " = Absorber(:", a.name, ", ", nameof(typeof(a.model)),
                        ", spectral_range_cm = ", spectral_range_cm(a.model), ")")
            end
        elseif name === :conditions && value !== nothing
            println(io, "conditions: ConditionsTable with ", length(value.times), " rows of ",
                    join(keys(value.values), ", "), " from ", first(value.times), " to ",
                    last(value.times), ", max_offset ", value.max_offset)
        elseif name in (:x_prior, :prior_sigma)
            println(io, name, ":")
            for (label, v) in zip(state_labels(value), value)
                println(io, "  ", label, " = ", repr(v))
            end
        elseif name === :retrieval
            println(io, "retrieval: RetrievalConfig")
            for f in fieldnames(typeof(value))
                println(io, "  ", f, " = ", repr(getfield(value, f)))
            end
        else
            println(io, name, " = ", repr(value))
        end
    end
    return String(take!(io))
end

export BatchRetrievalConfig, input_files, configuration_text
