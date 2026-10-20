"""
    SpectralFits.Apps

Workflows. Each app is a configuration struct plus `run(config)`, and reads its
configuration from a TOML file (see `read_config`). Apps use core and io, never each other.

- `Apps.BatchRetrieval`: retrieve every time average of many measurement files.
- `Apps.ProfileOSSE`: synthetic vertical-profile retrievals.
- `Apps.LookupTables`: build cross-section lookup tables.

`Apps.run_config(path)` runs every run of a configuration file with the app its `app` key
names; `Apps.main(ARGS)` is the command-line entry point (bin/run_config.jl).
"""
module Apps

using ..IOLayer: read_config
import ..CoreLayer: report

include("batch_retrieval/BatchRetrieval.jl")
include("profile_osse/ProfileOSSE.jl")
include("lookup_tables/LookupTables.jl")

const APPS = Dict("batch_retrieval" => (config = BatchRetrieval.batch_config, run = BatchRetrieval.run),
                  "profile_osse" => (config = ProfileOSSE.profile_osse_config, run = ProfileOSSE.run),
                  "lookup_table" => (config = LookupTables.lookup_table_config, run = LookupTables.run))

"""
    run_config(path; variants = nothing, dry_run = false) -> Vector

Read the configuration file `path` (`read_config`: `extends`, `[[variants]]`), build every
run's configuration first (so a mistake in any variant fails before anything runs), then run
them in order with the app named by `app` (`"batch_retrieval"`, `"profile_osse"` or
`"lookup_table"`). Returns
the apps' summaries. `variants` selects variants by name; `dry_run = true` only builds and
returns the configurations.

```julia
summaries = SpectralFits.Apps.run_config("configs/hit08.toml")
foreach(s -> print(report(s)), summaries)
```
"""
function run_config(path::AbstractString; variants = nothing, dry_run::Bool = false)
    tables = read_config(path; variants = variants)
    configs = map(tables) do t
        app = get(t, "app", nothing)
        app === nothing && throw(ArgumentError("$path: missing key \"app\" (one of $(sort(collect(keys(APPS)))))"))
        haskey(APPS, app) || throw(ArgumentError("$path: unknown app \"$app\"; apps are $(sort(collect(keys(APPS))))"))
        (app = app, config = APPS[app].config(t))
    end
    dry_run && return [c.config for c in configs]
    return [APPS[c.app].run(c.config) for c in configs]
end

"""
    main(args = ARGS)

Command line: `julia --project=<SpectralFit.jl> bin/run_config.jl <config.toml> [variant ...]
[--dry-run]`. Runs the configuration and prints each summary's report.
"""
function main(args = ARGS)
    if isempty(args) || args[1] in ("-h", "--help")
        println("usage: run_config.jl <config.toml> [variant ...] [--dry-run]")
        return nothing
    end
    dry = "--dry-run" in args
    rest = [a for a in args[2:end] if a != "--dry-run"]
    out = run_config(args[1]; variants = isempty(rest) ? nothing : rest, dry_run = dry)
    for s in out
        dry ? println("ok: ", s.experiment) : print(report(s))
    end
    return nothing
end

end # module Apps
