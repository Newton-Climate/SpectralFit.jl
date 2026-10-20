"""
    SpectralFits.Apps.ProfileOSSE

Synthetic vertical-profile retrievals (observing system simulation experiments): a
`ProfileOSSEConfig` (usually from a TOML file, `profile_osse_config`) plus
`ProfileOSSE.run(config)`, which returns a `ProfileOSSESummary`. `run` is not exported
because it would clash with `Base.run`.

```julia
import SpectralFits.Apps.ProfileOSSE
summary = ProfileOSSE.run(ProfileOSSE.profile_osse_config("configs/ch4_profile.toml"))
print(report(summary))
```
"""
module ProfileOSSE

using Dates
using Printf
using Random
using Statistics
using LinearAlgebra
using ComponentArrays
using ...CoreLayer
using ...IOLayer
import ...CoreLayer
import ...IOLayer: config_value, config_table, check_keys

include("profile_config.jl")
include("profile_osse.jl")

end # module ProfileOSSE
