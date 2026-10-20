"""
    SpectralFits.Apps.BatchRetrieval

Batch retrieval over many measurement files (phase 9 of docs/architecture_plan.md): a
`BatchRetrievalConfig` plus `BatchRetrieval.run(config)`, which returns a
`BatchRetrievalSummary` (see `report`). `run` is not exported because it would clash with
`Base.run`; import it explicitly:

```julia
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, run
```
"""
module BatchRetrieval

using Dates
using Distributed
using Printf
using Statistics
using ComponentArrays
using ...CoreLayer
using ...IOLayer
import ...CoreLayer

include("batch_config.jl")
include("batch_retrieval.jl")

end # module BatchRetrieval
