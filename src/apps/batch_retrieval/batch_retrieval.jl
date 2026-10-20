# Batch retrieval workflow (section 3.13, phase 9). Replaces v0.1 `process_all_files`,
# `run_inversion` and `fit_spectra`. For each input file: read (noise model applied to the
# full-spectrum records), time-average, retrieve every averaged measurement, save the results.
# A failure in one measurement or one file is logged and recorded; the batch continues.

const FileFailure = NamedTuple{(:file, :message),Tuple{String,String}}
const MeasurementOutcome = NamedTuple{(:source_file, :index, :time, :status, :exception, :message,
                                       :chi2_obs_per_dof, :dfs_total),
                                      Tuple{String,Int,DateTime,Symbol,Bool,String,Float64,Float64}}

"""
    BatchRetrievalSummary

What `BatchRetrieval.run` did. Fields:

- `experiment`, `run_folder`: the experiment name and the run folder created.
- `start_time`, `end_time`: local `DateTime`s.
- `files_processed`, `files_skipped`: input paths processed and skipped (pattern mismatch).
- `file_failures`: `(file, message)` for files that could not be read or averaged.
- `result_files`: one results file per processed input (see `save_results`).
- `outcomes`: one `NamedTuple` per averaged measurement with `source_file`, `index`,
  `time`, `status` (`:Converged`, `:MaxIterations` or `:Failed`), `exception` (true when
  the retrieval raised an exception rather than returning a result), `message`,
  `chi2_obs_per_dof` and `dfs_total` (`NaN` when not available).

`report(summary)` gives a plain-text summary.
"""
struct BatchRetrievalSummary
    experiment::String
    run_folder::String
    start_time::DateTime
    end_time::DateTime
    files_processed::Vector{String}
    files_skipped::Vector{String}
    file_failures::Vector{FileFailure}
    result_files::Vector{String}
    outcomes::Vector{MeasurementOutcome}
end

# ---------------------------------------------------------------- one file, one measurement

"""
    averaged_measurements(config::BatchRetrievalConfig, path) -> Vector{Measurement}

Read one DCS file with `config.reader` (with `config.noise`, when given, as the noise model
applied to the full-spectrum records) and average it into bins of `config.averaging_period`.

# Example
```julia
ms = averaged_measurements(config, "data/20160921.h5")
```
"""
function averaged_measurements(config::BatchRetrievalConfig, path::AbstractString)
    r = config.reader
    reader = config.noise === nothing ? r :
             DCSReaderConfig(r.path_length_cm, r.pressure_scale, r.field_layout, config.noise,
                             r.invalid_records, r.time_epoch)
    records = read_measurements(DCS, path; reader_config = reader)
    return time_average(records, config.averaging_period)
end

"""
    retrieve_measurement(config::BatchRetrievalConfig, measurement)
        -> (result = Union{Nothing,RetrievalResult}, message = String)

Build the problem with `build_problem` and solve it with `config.retrieval.solver`. An
exception (for example non-finite data rejected by `RetrievalProblem`) is caught and returned
as `result = nothing` with the error text in `message`; an `InterruptException` is rethrown.

# Example
```julia
outcome = retrieve_measurement(config, ms[1])
outcome.result === nothing && @warn outcome.message
```
"""
function retrieve_measurement(config::BatchRetrievalConfig, measurement::Measurement)
    try
        problem = build_problem(config.retrieval, config.absorbers, measurement, config.x_prior,
                                config.prior_sigma)
        result = solve(problem, config.retrieval.solver)
        return (result = result, message = result.message)
    catch err
        err isa InterruptException && rethrow()
        return (result = nothing, message = sprint(showerror, err))
    end
end

function _retrieve_all(config::BatchRetrievalConfig, measurements::Vector{<:Measurement})
    task = m -> retrieve_measurement(config, m)
    config.parallel === :serial && return map(task, measurements)
    nprocs() > 1 || @warn "parallel = :processes but no worker processes; running on process 1"
    return pmap(task, CachingPool(workers()), measurements)
end

# ---------------------------------------------------------------- run folder and manifest

function _git_hash()
    try
        dir = pkgdir(parentmodule(parentmodule(@__MODULE__)))
        cmd = Cmd(`git rev-parse --short HEAD`; dir = dir)
        return strip(read(pipeline(cmd; stderr = devnull), String))
    catch
        return "nogit"
    end
end

function _create_run_folder(config::BatchRetrievalConfig, start_time::DateTime, git_hash)
    base = joinpath(config.output_root,
                    string(Dates.format(start_time, "yyyy-mm-dd"), "_", config.experiment, "_",
                           git_hash))
    folder, k = base, 1
    while ispath(folder)                         # a second run on the same day and commit
        k += 1
        folder = string(base, "_", k)
    end
    mkpath(joinpath(folder, "results"))
    return folder
end

function _results_path(folder::AbstractString, input::AbstractString, used::Set{String})
    stem = splitext(basename(input))[1]
    name, k = string(stem, "_results.jld2"), 1
    while name in used
        k += 1
        name = string(stem, "_", k, "_results.jld2")
    end
    push!(used, name)
    return joinpath(folder, "results", name)
end

_toml_string(s) = string('"', escape_string(string(s), "\$"), '"')
_toml_value(v::AbstractString) = _toml_string(v)
_toml_value(v::Union{Integer,Bool}) = string(v)
_toml_value(v::AbstractVector) = string("[", join((_toml_value(e) for e in v), ", "), "]")
_toml_value(v) = _toml_string(v)

function _write_manifest(path, summary::BatchRetrievalSummary, config, git_hash)
    counts = outcome_counts(summary)
    entries = [
        "experiment" => summary.experiment,
        "run_folder" => abspath(summary.run_folder),
        "git_hash" => git_hash,
        "spectralfits_version" => string(pkgversion(parentmodule(parentmodule(@__MODULE__)))),
        "julia_version" => string(VERSION),
        "start_time" => string(summary.start_time),
        "end_time" => string(summary.end_time),
        "parallel" => string(config.parallel),
        "n_workers" => config.parallel === :serial ? 1 : nworkers(),
        "averaging_period" => string(config.averaging_period),
        "n_files_processed" => length(summary.files_processed),
        "n_files_skipped" => length(summary.files_skipped),
        "n_file_failures" => length(summary.file_failures),
        "n_measurements" => counts.total,
        "n_converged" => counts.converged,
        "n_max_iterations" => counts.max_iterations,
        "n_failed" => counts.failed,
        "inputs" => abspath.(summary.files_processed),
        "skipped" => abspath.(summary.files_skipped),
        "result_files" => abspath.(summary.result_files),
        "file_failures" => [string(f.file, ": ", f.message) for f in summary.file_failures],
    ]
    open(path, "w") do io
        println(io, "# SpectralFits batch retrieval manifest")
        for (key, value) in entries
            println(io, key, " = ", _toml_value(value))
        end
    end
    return path
end

# ---------------------------------------------------------------- run

"""
    BatchRetrieval.run(config::BatchRetrievalConfig) -> BatchRetrievalSummary

Run a batch retrieval:

1. Create the run folder `<output_root>/<yyyy-mm-dd>_<experiment>_<git short hash or
   "nogit">/` (suffix `_2`, `_3`, ... when it exists) and write `configuration.txt`
   (`configuration_text`).
2. For each input file matching `config.file_pattern` (others are logged and skipped): read
   it, apply the noise model to the full-spectrum records, `time_average` with
   `config.averaging_period`.
3. Retrieve every averaged measurement (`retrieve_measurement`; serially or with `pmap`
   over worker processes, as `config.parallel` says). An exception or a `Failed` status is
   logged with `@warn` and recorded; the batch continues.
4. Write `results/<input name>_results.jld2` per processed input with `save_results`: the
   results with their measurement times and metadata (plus `measurement_index`), and as
   attributes the source file, the number of averaged measurements and the measurements that
   raised an exception (`failures`: index, time, message). A file that could not be read
   gets a results file with no results and the error in `attributes["file_error"]`.
5. Write `manifest.toml` (inputs, skipped files, results files, versions, git hash, start
   and end time, counts).

Not exported, because `run` would clash with `Base.run`: call `BatchRetrieval.run(config)` or
import it explicitly.

# Example
```julia
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, run
summary = run(config)
print(report(summary))
```
"""
function run(config::BatchRetrievalConfig)
    start_time = now()
    git_hash = _git_hash()
    folder = _create_run_folder(config, start_time, git_hash)
    write(joinpath(folder, "configuration.txt"), configuration_text(config))
    files = input_files(config)
    for f in files.skipped
        @info "Skipping file (does not match file_pattern)" file = f pattern = config.file_pattern
    end
    @info "Batch retrieval started" experiment = config.experiment run_folder = folder n_files = length(files.selected) parallel = config.parallel

    # read and average every file first, so retrievals of all files share one pmap
    per_file = map(files.selected) do path
        try
            ms = averaged_measurements(config, path)
            @info "Read file" file = path n_measurements = length(ms)
            return (measurements = ms, error = "")
        catch err
            err isa InterruptException && rethrow()
            message = sprint(showerror, err)
            @warn "Could not read file; skipping it" file = path message
            return (measurements = Measurement[], error = message)
        end
    end
    all_measurements = [m for f in per_file for m in f.measurements]
    retrieved = isempty(all_measurements) ? [] : _retrieve_all(config, all_measurements)

    outcomes = MeasurementOutcome[]
    file_failures = FileFailure[]
    result_files = String[]
    used_names = Set{String}()
    offset = 0
    for (path, f) in zip(files.selected, per_file)
        isempty(f.error) || push!(file_failures, (file = path, message = f.error))
        n = length(f.measurements)
        file_outcomes = retrieved[offset+1:offset+n]
        offset += n
        results = RetrievalResult[]
        times, metadata, failures = DateTime[], NamedTuple[], NamedTuple[]
        for (i, (m, o)) in enumerate(zip(f.measurements, file_outcomes))
            status = o.result === nothing ? :Failed : Symbol(string(o.result.status))
            if o.result === nothing
                push!(failures, (index = i, time = m.time, message = o.message))
            else
                push!(results, o.result)
                push!(times, m.time)
                push!(metadata, merge(m.metadata, (measurement_index = i,)))
            end
            status === :Failed &&
                @warn "Retrieval failed" file = path index = i time = m.time message = o.message
            push!(outcomes, (source_file = String(path), index = i, time = m.time,
                             status = status, exception = o.result === nothing,
                             message = String(o.message),
                             chi2_obs_per_dof = o.result === nothing ? NaN : o.result.chi2_obs_per_dof,
                             dfs_total = o.result === nothing ? NaN : o.result.dfs_total))
        end
        out = _results_path(folder, path, used_names)
        attributes = (source_file = abspath(path), experiment = config.experiment,
                      n_measurements = n, failures = failures, file_error = f.error)
        save_results(out, results; times = times, metadata = metadata,
                     attributes = attributes)
        push!(result_files, out)
    end

    summary = BatchRetrievalSummary(config.experiment, folder, start_time, now(),
                                    copy(files.selected), copy(files.skipped), file_failures,
                                    result_files, outcomes)
    _write_manifest(joinpath(folder, "manifest.toml"), summary, config, git_hash)
    counts = outcome_counts(summary)
    @info "Batch retrieval finished" run_folder = folder counts.converged counts.max_iterations counts.failed
    return summary
end

# ---------------------------------------------------------------- report

"""
    outcome_counts(summary::BatchRetrievalSummary)
        -> (total, converged, max_iterations, failed, exceptions)

Number of averaged measurements by outcome. `failed` counts both `Failed` statuses and
exceptions; `exceptions` counts the latter alone.

# Example
```julia
outcome_counts(summary).converged
```
"""
function outcome_counts(summary::BatchRetrievalSummary)
    o = summary.outcomes
    return (total = length(o), converged = count(x -> x.status === :Converged, o),
            max_iterations = count(x -> x.status === :MaxIterations, o),
            failed = count(x -> x.status === :Failed, o),
            exceptions = count(x -> x.exception, o))
end

function _distribution_line(io, name, values)
    v = filter(isfinite, values)
    if isempty(v)
        println(io, rpad(name, 18), " no finite values")
    else
        q = quantile(v, [0.0, 0.25, 0.5, 0.75, 1.0])
        @printf(io, "%-18s n=%d min=%.4g p25=%.4g median=%.4g p75=%.4g max=%.4g\n",
                name, length(v), q...)
    end
end

"""
    report(summary::BatchRetrievalSummary) -> String

Plain-text summary of a batch run: run folder, files processed, skipped and failed, counts of
converged, max-iterations and failed retrievals, and the distribution (min, quartiles,
max) of chi2 per degree of freedom (observation term) and of total DFS over the retrievals
that did not fail. Printing it is the caller's choice.

# Example
```julia
print(report(summary))
```
"""
function CoreLayer.report(summary::BatchRetrievalSummary)
    counts = outcome_counts(summary)
    io = IOBuffer()
    println(io, "Batch retrieval report: ", summary.experiment)
    println(io, "run_folder: ", summary.run_folder)
    println(io, "start: ", summary.start_time, "  end: ", summary.end_time)
    println(io, "files: processed=", length(summary.files_processed), " skipped=",
            length(summary.files_skipped), " unreadable=", length(summary.file_failures))
    println(io, "retrievals: total=", counts.total, " converged=", counts.converged,
            " max_iterations=", counts.max_iterations, " failed=", counts.failed,
            " (exceptions=", counts.exceptions, ")")
    ok = [o for o in summary.outcomes if o.status !== :Failed]
    _distribution_line(io, "chi2_obs_per_dof", [o.chi2_obs_per_dof for o in ok])
    _distribution_line(io, "dfs_total", [o.dfs_total for o in ok])
    for o in summary.outcomes
        o.status === :Failed &&
            println(io, "failed: ", basename(o.source_file), " #", o.index, " ", o.time, ": ",
                    first(o.message, 200))
    end
    for f in summary.file_failures
        println(io, "unreadable: ", f.file, ": ", first(f.message, 200))
    end
    return String(take!(io))
end

Base.show(io::IO, ::MIME"text/plain", summary::BatchRetrievalSummary) = print(io, report(summary))

export BatchRetrievalSummary, averaged_measurements, retrieve_measurement, outcome_counts
