module SpectralFits
include("constants.jl")
include("types.jl")
include("utils.jl")
include("read_data.jl")
include("forward_model.jl")
include("inversion.jl")
using OrderedCollections

### export our structs and types
# spectral-related types 
export MolecularMetaData, Molecule, Spectra, OrderedDict


# Dataset-related types
export Dataset, FrequencyComb
export FrequencyCombDataset, TimeAveragedFrequencyCombDataset

# measurement-related types
export Measurement, FrequencyCombMeasurement


### export spectral-related functions
export get_molecule_info, calculate_cross_sections, calculate_cross_sections!
export construct_spectra, construct_spectra!
export construct_spectra_by_layer, construct_spectra_by_layer!

# Dataset Processing functions
export read_DCS_data, take_time_average, get_measurement

# forward model functions
export calc_transmission, apply_instrument, calc_polynomial_term
export generate_forward_model, generate_profile_model


### fitting routines
export nonlinear_inversion, fit_spectra, run_inversion
export process_all_files

# some useful funcs fom utils.jl
export assemble_state_vector!, OCO_spectra
export make_vcd_profile, calc_gain_matrix

end # module
