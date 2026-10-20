"""
    SpectralFits.CoreLayer

Pure science: no file input or output, no printing, no global state. Every file in this
directory is included here in dependency order; each file declares its own exports.
"""
module CoreLayer

using LinearAlgebra
using SparseArrays
using Statistics
using Dates
using Printf
using Random
using ComponentArrays
import ForwardDiff
import DiffResults
import Interpolations
import vSmartMOM
import vSmartMOM.Absorption as Absorption

export ComponentVector

include("physical_constants.jl")
include("parameters.jl")
include("interfaces.jl")
include("state_vector.jl")
include("air_columns.jl")
include("measurements.jl")
include("noise_estimation.jl")

include("spectroscopy/precomputed.jl")
include("spectroscopy/cross_section_models.jl")
include("spectroscopy/cross_section_wrappers.jl")
include("spectroscopy/tabulation.jl")

include("atmospheres.jl")
include("baselines.jl")

include("instruments/interpolation.jl")
include("instruments/digital_filters.jl")
include("instruments/dcs.jl")
include("instruments/fts.jl")
include("instruments/grating.jl")
include("instruments/wms_laser.jl")
include("instruments/wms_demodulation.jl")
include("instruments/wms_observables.jl")
include("instruments/wms_noise.jl")

include("covariance_precision.jl")
include("diagnostics.jl")
include("retrieval_result.jl")
include("retrieval_problem.jl")
include("solvers.jl")

include("forward_model.jl")
include("forward_model_builder.jl")
include("retrieval_config.jl")
include("prior_covariance.jl")
include("synthetic_measurements.jl")

end # module CoreLayer
