# Run an app configuration file:
#     julia --project=<path to SpectralFit.jl> <path to SpectralFit.jl>/bin/run_config.jl config.toml [variant ...] [--dry-run]
using SpectralFits
SpectralFits.Apps.main(ARGS)
