# Retrieval results files (section 3.13, phase 8). The file holds only plain data (numbers,
# strings, arrays of numbers or strings, and Dict{String,Any} of those), so it can be read
# without SpectralFits and does not break when a struct layout changes.
#
# File layout (JLD2, top-level keys):
#
#   "format"          "SpectralFits.retrieval_results"
#   "format_version"  1
#   "attributes"      Dict{String,Any}: file-level plain data from the caller
#   "results"         Vector{Dict{String,Any}}, one per retrieval, keys:
#       "state_labels"                Vector{String}, `labels(x)` (e.g. "baseline.window_1[2]")
#       "x", "x_prior", "dfs_per_element"   Vector{Float64}, flat, in the order of the labels
#       "status"                      "Converged", "MaxIterations" or "Failed"
#       "message"                     String
#       "n_iterations"                Int
#       "rodgers_d2_per_state", "dfs_total", "chi2_obs_per_dof", "chi2_total_per_dof",
#       "prior_influence", "condition_number", "rmse"    Float64
#       "cost_history", "y", "y_model"           Vector{Float64}
#       "K", "posterior_covariance", "averaging_kernel"   Matrix{Float64}
#       "obs_precision", "prior_precision"       Matrix{Float64}, or Vector{Float64} of the
#                                                diagonal when "<name>_is_diagonal" is true
#       "obs_precision_covariance_rows", "..._cols", "..._values", "..._size"
#                                     sparse triplets (Int, Int, Float64) and (n, n) of the
#                                     covariance, present only when "obs_precision_is_covariance"
#                                     is true (a `CovariancePrecision`; "obs_precision" is then
#                                     an empty matrix)
#       "time"                        ISO 8601 string ("" when not given)
#       "coordinate"                  Vector{Float64} of the observations (wavenumber in cm-1, or
#                                     time in s); empty when not given
#       "metadata"                    Dict{String,Any}: the measurement's metadata

const RESULTS_FORMAT = "SpectralFits.retrieval_results"
const RESULTS_FORMAT_VERSION = 1

# ---------------------------------------------------------------- plain-data conversion

"""
    plain_value(v)

Convert `v` to plain data that JLD2 stores without any package type: numbers, strings, bools
and arrays of them are kept; `Symbol`, `DateTime`, `Dates.Period` and anything else become
strings; `NamedTuple` and `AbstractDict` become `Dict{String,Any}` (recursively); `nothing`
becomes `""`.
"""
plain_value(v::Union{Real,AbstractString}) = v isa AbstractString ? String(v) : v
plain_value(v::DateTime) = string(v)
plain_value(::Nothing) = ""
plain_value(v::Union{NamedTuple,AbstractDict}) =
    Dict{String,Any}(string(k) => plain_value(val) for (k, val) in pairs(v))
plain_value(v::AbstractArray) = eltype(v) <: Real ? collect(v) :
                                eltype(v) <: AbstractString ? String.(v) :
                                Any[plain_value(e) for e in v]
plain_value(v) = string(v)

function _store_precision!(d::Dict{String,Any}, name::String, P::AbstractMatrix)
    is_diagonal = P isa Diagonal
    d[name * "_is_diagonal"] = is_diagonal
    d[name] = is_diagonal ? Vector{Float64}(diag(P)) : Matrix{Float64}(P)
    return d
end

function _store_precision!(d::Dict{String,Any}, name::String, P::CovariancePrecision)
    rows, cols, values = findnz(observation_covariance(P))
    d[name * "_is_diagonal"] = false
    d[name * "_is_covariance"] = true
    d[name] = zeros(0, 0)
    d[name * "_covariance_rows"] = rows
    d[name * "_covariance_cols"] = cols
    d[name * "_covariance_values"] = values
    d[name * "_covariance_size"] = collect(size(P))
    return d
end

function _load_precision(d, name)
    if get(d, name * "_is_covariance", false)
        n, m = d[name * "_covariance_size"]
        return CovariancePrecision(sparse(Vector{Int}(d[name * "_covariance_rows"]),
                                          Vector{Int}(d[name * "_covariance_cols"]),
                                          Vector{Float64}(d[name * "_covariance_values"]), n, m))
    end
    return d[name * "_is_diagonal"] ? Diagonal(Vector{Float64}(d[name])) : Matrix{Float64}(d[name])
end

function _result_dict(r::RetrievalResult, time, metadata, coordinate)
    d = Dict{String,Any}(
        "state_labels" => state_labels(r.x),
        "x" => Vector{Float64}(collect(r.x)),
        "x_prior" => Vector{Float64}(collect(r.x_prior)),
        "dfs_per_element" => Vector{Float64}(collect(r.dfs_per_element)),
        "status" => string(r.status),
        "message" => r.message,
        "n_iterations" => r.n_iterations,
        "rodgers_d2_per_state" => r.rodgers_d2_per_state,
        "cost_history" => copy(r.cost_history),
        "y" => copy(r.y),
        "y_model" => copy(r.y_model),
        "K" => copy(r.K),
        "posterior_covariance" => copy(r.posterior_covariance),
        "averaging_kernel" => copy(r.averaging_kernel),
        "dfs_total" => r.dfs_total,
        "chi2_obs_per_dof" => r.chi2_obs_per_dof,
        "chi2_total_per_dof" => r.chi2_total_per_dof,
        "prior_influence" => r.prior_influence,
        "condition_number" => r.condition_number,
        "rmse" => r.rmse,
        "time" => time === nothing ? "" : string(time),
        "coordinate" => coordinate === nothing ? Float64[] : collect(Float64, coordinate),
        "metadata" => plain_value(metadata),
    )
    _store_precision!(d, "obs_precision", r.obs_precision)
    _store_precision!(d, "prior_precision", r.prior_precision)
    return d
end

# ---------------------------------------------------------------- state from labels

# "baseline.window_1[2]" -> (["baseline", "window_1"], (2,)); "CH4" -> (["CH4"], ())
function _parse_label(label::AbstractString)
    m = match(r"^([^\[]+)(?:\[([0-9, ]+)\])?$", label)
    m === nothing && throw(ArgumentError("cannot parse state label \"$label\""))
    path = String.(split(m.captures[1], '.'))
    index = m.captures[2] === nothing ? () : Tuple(parse.(Int, split(m.captures[2], ',')))
    return path, index
end

# Nested NamedTuple from entries (path, index, value) sharing the current level.
function _named_tuple_from_entries(entries)
    names = unique(e[1][1] for e in entries)
    values = map(names) do name
        group = [e for e in entries if e[1][1] == name]
        if all(e -> length(e[1]) == 1, group)
            if length(group) == 1 && isempty(group[1][2])
                group[1][3]                                   # scalar
            else
                dims = Tuple(maximum(e[2][k] for e in group) for k in eachindex(group[1][2]))
                reshape([e[3] for e in group], dims)          # column-major, as labels()
            end
        else
            _named_tuple_from_entries([(e[1][2:end], e[2], e[3]) for e in group])
        end
    end
    return NamedTuple{Tuple(Symbol.(names))}(Tuple(values))
end

"""
    state_from_labels(values, labels) -> ComponentVector

Rebuild a `ComponentVector` from its flat `values` and `labels` (as returned by
`ComponentArrays.labels` or `state_labels`). Supports scalars, vectors, matrices and nested
components. Throws an `ArgumentError` when the labels do not describe the values exactly.

# Example
```julia
x = state_from_labels([1.0, 2.0, 3.0], ["CH4", "baseline.window_1[1]", "baseline.window_1[2]"])
x.baseline.window_1          # [2.0, 3.0]
```
"""
function state_from_labels(values::AbstractVector{<:Real}, labels::AbstractVector{<:AbstractString})
    length(values) == length(labels) ||
        throw(ArgumentError("state_from_labels: $(length(values)) values for $(length(labels)) labels"))
    isempty(values) && return ComponentVector{Float64}()
    entries = [(_parse_label(l)..., Float64(v)) for (l, v) in zip(labels, values)]
    x = ComponentVector(_named_tuple_from_entries(entries))
    state_labels(x) == collect(labels) ||
        throw(ArgumentError("state_from_labels: labels are not in ComponentVector order"))
    return x
end

# ---------------------------------------------------------------- save and load

"""
    save_results(path, results; times = nothing, metadata = nothing, coordinates = nothing,
                 attributes = Dict{String,Any}()) -> path

Write retrieval results to a JLD2 file holding only plain data (layout in the header of
`src/io/retrieval_results_jld2.jl`).

- `results`: vector of `RetrievalResult` (may be empty).
- `times`: `nothing` or one `DateTime` per result (the measurement time).
- `metadata`: `nothing` or one `NamedTuple`/`Dict` per result (for example the measurement's
  metadata: source file, record index, number averaged); converted with `plain_value`.
- `coordinates`: `nothing` or one vector per result, the coordinate of its observations in
  the order of `y` (wavenumber in cm-1 for spectra, time in s for WMS).
- `attributes`: file-level `NamedTuple`/`Dict` (for example the source file and failed
  measurements); converted with `plain_value`.

An existing file at `path` is replaced.

# Example
```julia
save_results("20160921_results.jld2", results;
             times = [m.time for m in measurements],
             metadata = [m.metadata for m in measurements],
             attributes = (source_file = "20160921.h5",))
```
"""
function save_results(path::AbstractString, results::AbstractVector{<:RetrievalResult};
                      times = nothing, metadata = nothing, coordinates = nothing,
                      attributes = Dict{String,Any}())
    n = length(results)
    times === nothing || length(times) == n ||
        throw(ArgumentError("times: has $(length(times)) entries for $n results"))
    metadata === nothing || length(metadata) == n ||
        throw(ArgumentError("metadata: has $(length(metadata)) entries for $n results"))
    coordinates === nothing || length(coordinates) == n ||
        throw(ArgumentError("coordinates: has $(length(coordinates)) entries for $n results"))
    for i in 1:(coordinates === nothing ? 0 : n)
        length(coordinates[i]) == length(results[i].y) || throw(DimensionMismatch(
            "coordinates[$i]: has $(length(coordinates[i])) values, result $i has $(length(results[i].y)) observations"))
    end
    dicts = Dict{String,Any}[_result_dict(results[i],
                                          times === nothing ? nothing : times[i],
                                          metadata === nothing ? (;) : metadata[i],
                                          coordinates === nothing ? nothing : coordinates[i])
                             for i in 1:n]
    jldopen(path, "w") do file
        file["format"] = RESULTS_FORMAT
        file["format_version"] = RESULTS_FORMAT_VERSION
        file["attributes"] = plain_value(attributes)
        file["results"] = dicts
    end
    return path
end

"""
    load_results(path) -> (results = Vector{NamedTuple}, attributes = Dict{String,Any})

Read a file written by `save_results`. Each result is a `NamedTuple` with the field names of
`RetrievalResult` plus `state_labels`, `time` (`DateTime`, or `nothing` when not saved),
`coordinate` (`Vector{Float64}`, empty when not saved) and `metadata` (`Dict{String,Any}`):

- `x`, `x_prior`, `dfs_per_element`: `ComponentVector`s rebuilt from the stored flat vectors
  and labels with `state_from_labels`, so `r.x.CH4` works as on the original result.
- `status`: a `RetrievalStatus` (`Converged`, `MaxIterations`, `Failed`).
- `obs_precision`, `prior_precision`: `Diagonal` when saved from a `Diagonal`, a
  `CovariancePrecision` when saved from one, else `Matrix`.
- every other field: the stored number, string, vector or matrix.

# Example
```julia
loaded = load_results("20160921_results.jld2")
[r.x.CH4 for r in loaded.results if r.status == Converged]
```
"""
function load_results(path::AbstractString)
    format, version, attributes, dicts = jldopen(path, "r") do file
        read_or(key, default) = haskey(file, key) ? file[key] : default
        (read_or("format", ""), read_or("format_version", 0),
         read_or("attributes", Dict{String,Any}()), read_or("results", Dict{String,Any}[]))
    end
    format == RESULTS_FORMAT ||
        throw(ArgumentError("$path: not a SpectralFits results file (format \"$format\")"))
    version <= RESULTS_FORMAT_VERSION ||
        throw(ArgumentError("$path: format_version $version is newer than this reader"))
    return (results = [_result_from_dict(d) for d in dicts], attributes = attributes)
end

const STATUS_BY_NAME = Dict(string(s) => s for s in instances(RetrievalStatus))

function _result_from_dict(d::AbstractDict)
    labels = Vector{String}(d["state_labels"])
    time = isempty(d["time"]) ? nothing : DateTime(d["time"])
    return (x = state_from_labels(d["x"], labels),
            x_prior = state_from_labels(d["x_prior"], labels),
            status = STATUS_BY_NAME[d["status"]],
            message = String(d["message"]),
            n_iterations = Int(d["n_iterations"]),
            rodgers_d2_per_state = Float64(d["rodgers_d2_per_state"]),
            cost_history = Vector{Float64}(d["cost_history"]),
            y = Vector{Float64}(d["y"]),
            y_model = Vector{Float64}(d["y_model"]),
            K = Matrix{Float64}(d["K"]),
            obs_precision = _load_precision(d, "obs_precision"),
            prior_precision = _load_precision(d, "prior_precision"),
            posterior_covariance = Matrix{Float64}(d["posterior_covariance"]),
            averaging_kernel = Matrix{Float64}(d["averaging_kernel"]),
            dfs_per_element = state_from_labels(d["dfs_per_element"], labels),
            dfs_total = Float64(d["dfs_total"]),
            chi2_obs_per_dof = Float64(d["chi2_obs_per_dof"]),
            chi2_total_per_dof = Float64(d["chi2_total_per_dof"]),
            prior_influence = Float64(d["prior_influence"]),
            condition_number = Float64(d["condition_number"]),
            rmse = Float64(get(d, "rmse", NaN)),
            state_labels = labels,
            time = time,
            coordinate = Vector{Float64}(get(d, "coordinate", Float64[])),
            metadata = d["metadata"])
end

export save_results, load_results, state_from_labels, plain_value
