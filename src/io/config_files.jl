# Configuration files for the apps (TOML). One file describes one experiment, or several
# variants of it; shared settings live in a base file named by `extends`.
#
#   app = "batch_retrieval"            # which app runs it (see SpectralFits.Apps.run_config)
#   experiment = "hit08"
#   extends = "campaign_2016.toml"     # optional; tables are merged, this file wins
#   ...
#   [[variants]]                       # optional; one run per variant
#   name = "ch4"                       # experiment becomes "hit08_ch4"
#   retrieval.window_cm = [6050.0, 6115.0]
#   absorbers.CO2 = { _remove = true } # markers: _remove a key, _replace a table (merge_config)
#
# Relative paths are resolved against the directory of the file that sets them. A key holds
# a path when its name is `file`, `files`, `extends`, or ends in `_file`, `_files`,
# `_directory` or `_root`.

"""Whether a configuration key names a path (see the header of io/config_files.jl)."""
is_path_key(key::AbstractString) =
    key in ("file", "files", "extends") || any(s -> endswith(key, s), ("_file", "_files", "_directory", "_root"))

resolve_path(dir, p::AbstractString) = isabspath(p) ? p : normpath(joinpath(dir, expanduser(p)))
resolve_path(dir, p::AbstractVector) = [resolve_path(dir, q) for q in p]
resolve_path(dir, p) = p

"""Copy of `d` with every path-valued entry made absolute against `dir` (nested tables too)."""
function resolve_paths(d::AbstractDict, dir::AbstractString)
    out = Dict{String,Any}()
    for (k, v) in d
        out[k] = if v isa AbstractDict
            resolve_paths(v, dir)
        elseif v isa AbstractVector && !isempty(v) && all(e -> e isa AbstractDict, v)
            Any[resolve_paths(e, dir) for e in v]
        elseif is_path_key(k)
            resolve_path(dir, v)
        else
            v
        end
    end
    return out
end

"""
    merge_config(base, override) -> Dict{String,Any}

Recursive merge of two configuration tables: tables are merged key by key, every other value
(including arrays) in `override` replaces the one in `base`. Two markers in an override
table change that:

- `_replace = true`: the table replaces the base table instead of being merged into it,
  for example `absorbers.CH4 = { _replace = true, file = "ch4.jld2" }` swaps a line list
  (with its `molecule` and `range_cm`) for a lookup table;
- `_remove = true`: the key is removed, for example `absorbers.CO2 = { _remove = true }`.
"""
function merge_config(base::AbstractDict, override::AbstractDict)
    out = Dict{String,Any}(base)
    for (k, v) in override
        if v isa AbstractDict && get(v, "_remove", false) === true
            delete!(out, k)
        elseif v isa AbstractDict && get(v, "_replace", false) === true
            out[k] = merge_config(Dict{String,Any}(), Dict{String,Any}(kk => vv for (kk, vv) in v if kk != "_replace"))
        elseif v isa AbstractDict
            out[k] = merge_config(get(out, k, nothing) isa AbstractDict ? out[k] : Dict{String,Any}(), v)
        else
            out[k] = v
        end
    end
    return out
end

function _read_config_file(path::AbstractString, seen::Vector{String})
    path = abspath(path)
    isfile(path) || throw(ArgumentError("configuration file not found: $path"))
    path in seen && throw(ArgumentError("configuration files extend each other in a loop: $(join([seen; path], " -> "))"))
    d = resolve_paths(TOML.parsefile(path), dirname(path))
    base = pop!(d, "extends", nothing)
    base === nothing && return merge_config(Dict{String,Any}(), d)   # drops stray markers
    parent = _read_config_file(base, [seen; path])
    pop!(parent, "variants", nothing)              # variants are not inherited
    return merge_config(parent, d)
end

"""
    read_config(path; variants = nothing) -> Vector{Dict{String,Any}}

Read an app configuration file: follow `extends`, resolve relative paths, and expand
`[[variants]]`. Returns one configuration table per run: the file itself when it has no
variants, otherwise one per variant (the variant merged over the file, `experiment` set to
`<experiment>_<variant name>`, and `variant` set to the variant name). `variants` (a vector
of names) keeps only those. Each table also gets `config_file`, the absolute path read.

```julia
configs = read_config("configs/hit08.toml")          # two runs: hit08_ch4, hit08_co2
configs = read_config("configs/hit24.toml"; variants = ["ch4_hit24"])
```
"""
function read_config(path::AbstractString; variants = nothing)
    d = _read_config_file(path, String[])
    d["config_file"] = abspath(path)
    haskey(d, "experiment") || throw(ArgumentError("$path: missing key \"experiment\""))
    specs = pop!(d, "variants", nothing)
    if specs === nothing
        variants === nothing || throw(ArgumentError("$path has no [[variants]]; cannot select $variants"))
        return [d]
    end
    names = String[]
    configs = Dict{String,Any}[]
    for v in specs
        haskey(v, "name") || throw(ArgumentError("$path: every [[variants]] entry needs a \"name\""))
        name = String(v["name"])
        name in names && throw(ArgumentError("$path: variant name \"$name\" is used twice"))
        push!(names, name)
        c = merge_config(d, Dict{String,Any}(k => x for (k, x) in v if k != "name"))
        c["experiment"] = string(d["experiment"], "_", name)
        c["variant"] = name
        push!(configs, c)
    end
    variants === nothing && return configs
    unknown = setdiff(String.(variants), names)
    isempty(unknown) || throw(ArgumentError("$path: no variants named $unknown; variants are $names"))
    return [c for c in configs if c["variant"] in variants]
end

"""
    write_config(path, config::AbstractDict) -> path

Write a resolved configuration table as TOML (the apps copy it into each run folder).
"""
function write_config(path::AbstractString, config::AbstractDict)
    open(io -> TOML.print(io, config; sorted = true), path, "w")
    return path
end

# ---------------------------------------------------------------- reading values

"""
    config_value(d, key, default = REQUIRED; where = "") -> value

`d[key]`, or `default` when absent; a missing required key is an `ArgumentError` naming the
key and the table (`where`).
"""
struct Required end
const REQUIRED = Required()
function config_value(d::AbstractDict, key::AbstractString, default = REQUIRED; where::AbstractString = "")
    haskey(d, key) && return d[key]
    default isa Required && throw(ArgumentError("configuration$(isempty(where) ? "" : " [$where]"): missing key \"$key\""))
    return default
end

"""`config_table(d, key)`: the sub-table `d[key]`, or an empty table when absent."""
function config_table(d::AbstractDict, key::AbstractString; where::AbstractString = "")
    v = get(d, key, Dict{String,Any}())
    v isa AbstractDict || throw(ArgumentError("configuration$(isempty(where) ? "" : " [$where]"): \"$key\" must be a table"))
    return v
end

"""Check that `d` has no keys outside `allowed`, so misspelled settings are not ignored."""
function check_keys(d::AbstractDict, allowed, where::AbstractString)
    unknown = setdiff(collect(keys(d)), allowed)
    isempty(unknown) || throw(ArgumentError(
        "configuration [$where]: unknown keys $(sort(unknown)); allowed: $(sort(collect(allowed)))"))
    return nothing
end

"""`(lo, hi)` pairs from a TOML array of two-element arrays."""
windows_from_config(v) = Tuple{Float64,Float64}[(Float64(w[1]), Float64(w[2])) for w in v]

export read_config, write_config, merge_config
