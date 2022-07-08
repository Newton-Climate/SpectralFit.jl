module SpectralFits

## import packages
# for math calculations
using LinearAlgebra, ForwardDiff, DiffResults
using Statistics, Interpolations

# parallel processing
using Distributed

# File IO
using Dates, OrderedCollections, HDF5, JLD2, ProgressMeter

# Import the Radiative Transfer Code
using vSmartMOM, vSmartMOM.Absorption, vSmartMOM.Architectures

## create name-space
include("constants.jl") # physical constants
include("types.jl") # custom datatypes and structs 
include("utils.jl") # helper-functions 
include("spectroscopy.jl") # for calculating absorption cross-sections and MolecularMetaData type 
include("read_data.jl") # file IO and creating AbstractDataset and AbstractMeasurements
include("forward_model.jl") # functions for creating a forward model
include("inversion.jl") # functions for retrieving and fitting 

### export our structs and types
# spectral-related types 
export MolecularMetaData, Molecule, Spectra, OrderedDict
export AbstractResults, InversionResults, FailedInversion
export setup_molecules


# Dataset-related types
export AbstractDataset, FrequencyCombDataset

# measurement-related types
export AbstractMeasurement, FrequencyCombMeasurement

# State Vector types
export StateVector, ProfileStateVector, RetrievalSetup

### export spectral-related functions
export get_molecule_info, calculate_cross_sections, calculate_cross_sections!
export construct_spectra, construct_spectra!
export construct_spectra_by_layer, construct_spectra_by_layer!

# Dataset Processing functions
export read_DCS_data, take_time_average!, get_measurement

# forward model functions
export calc_transmission!, apply_instrument, calc_polynomial_term
export generate_forward_model, generate_profile_model


### fitting routines
export nonlinear_inversion, profile_inversion, fit_spectra, run_inversion
export process_all_files

# some useful funcs fom utils.jl
export assemble_state_vector!, prior_shape_params
export make_vcd_profile, calc_gain_matrix

# certain packages visible to the user
export Dates, OrderedCollections, Statistics, JLD2
export vSmartMOM

end # module
