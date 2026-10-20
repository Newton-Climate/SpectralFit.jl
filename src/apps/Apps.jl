"""
    SpectralFits.Apps

Workflows. Each app is a configuration struct plus `run(config)`. Apps use core and io,
never each other.
"""
module Apps

include("batch_retrieval/BatchRetrieval.jl")

end # module Apps
