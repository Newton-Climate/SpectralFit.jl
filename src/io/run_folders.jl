# Run folders shared by the apps: <output_root>/<yyyy-mm-dd>_<experiment>_<git short hash>[_k]/

"""
    git_short_hash() -> String

Short git commit hash of the SpectralFits source tree, or `"nogit"` when it is not a git
checkout. Recorded in run folder names and manifests.
"""
function git_short_hash()
    try
        dir = pkgdir(parentmodule(@__MODULE__))
        cmd = Cmd(`git rev-parse --short HEAD`; dir = dir)
        return String(strip(read(pipeline(cmd; stderr = devnull), String)))
    catch
        return "nogit"
    end
end

"""
    create_run_folder(output_root, experiment; time = now(), git_hash = git_short_hash()) -> String

Create and return `<output_root>/<yyyy-mm-dd>_<experiment>_<git_hash>/` with a `results/`
subfolder; a second run on the same day and commit gets the suffix `_2`, `_3`, ...
`experiment` may contain only letters, digits, `-`, `_` and `.`.
"""
function create_run_folder(output_root::AbstractString, experiment::AbstractString;
                           time::DateTime = now(), git_hash::AbstractString = git_short_hash())
    occursin(r"^[A-Za-z0-9._-]+$", experiment) || throw(ArgumentError(
        "experiment: \"$experiment\" may contain only letters, digits, '.', '_' and '-'"))
    base = joinpath(output_root, string(Dates.format(time, "yyyy-mm-dd"), "_", experiment, "_", git_hash))
    folder, k = base, 1
    while ispath(folder)
        k += 1
        folder = string(base, "_", k)
    end
    mkpath(joinpath(folder, "results"))
    return folder
end

const RUN_FOLDER_PATTERN = r"^\d{4}-\d{2}-\d{2}_(.+)_([0-9a-f]{7,40}|nogit)(_\d+)?$"

"""
    find_run(output_root, experiment; require_results = true) -> String

The most recent run folder of `experiment` in `output_root` (made by `create_run_folder`),
by modification time. With `require_results`, folders whose `results/` holds no file (an
aborted run) are passed over. Throws an `ArgumentError` when there is none.

```julia
folder = find_run("../data/runs", "merlin")
loaded = load_results(joinpath(folder, "results", "20160921_results.jld2"))
```
"""
function find_run(output_root::AbstractString, experiment::AbstractString; require_results::Bool = true)
    folders = String[]
    if isdir(output_root)
        for d in readdir(output_root)
            m = match(RUN_FOLDER_PATTERN, d)
            folder = joinpath(output_root, d)
            m !== nothing && m.captures[1] == experiment && isdir(folder) || continue
            require_results && !_has_results(folder) && continue
            push!(folders, folder)
        end
    end
    isempty(folders) && throw(ArgumentError("no run folder$(require_results ? " with results" : "") for experiment \"$experiment\" in $output_root"))
    return folders[argmax(mtime.(folders))]
end

_has_results(folder) = (r = joinpath(folder, "results"); isdir(r) && any(f -> isfile(joinpath(r, f)), readdir(r)))

"""
    run_experiments(output_root) -> Vector{String}

Sorted names of the experiments that have run folders in `output_root`.
"""
function run_experiments(output_root::AbstractString)
    isdir(output_root) || return String[]
    names = [m.captures[1] for m in (match(RUN_FOLDER_PATTERN, d) for d in readdir(output_root)) if m !== nothing]
    return sort(unique(names))
end

export git_short_hash, create_run_folder, find_run, run_experiments
