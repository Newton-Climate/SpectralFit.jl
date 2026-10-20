"""
    SpectralFits.IOLayer

Every file read and write: instrument data readers, line-list and lookup-table loaders,
results files. Uses core types; core never uses this module.
"""
module IOLayer

using Dates
using LinearAlgebra
using HDF5
using JLD2
using ComponentArrays
import vSmartMOM
import vSmartMOM.Absorption as Absorption
import Interpolations
using ..CoreLayer

include("dcs_hdf5.jl")
include("line_lists.jl")
include("absco_tables.jl")
include("retrieval_results_jld2.jl")

end # module IOLayer
