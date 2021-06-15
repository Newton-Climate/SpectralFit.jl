module SpectralFits

### export our structs and types
# spectral-related types 
export MolecularMetaData, Molecule, Spectra


# Dataset-related types
export Dataset, FrequencyComb
export FrequencyCombDataset, TimeAveragedFrequencyCombDataset

# measurement-related types
export Measurement, FrequencyCombMeasurement


### export spectral-related functions
export get_molecule_info, calculate_cross_sections, calculate_cross_sections!
export construct_spectra, construct_spectra!

# Dataset Processing functions
export read_DCS_data, take_time_average, get_measurement

# forward model functions
export calc_transmission, apply_instrument, calc_polynomial_term
export generate_forward_model
    
### fitting routines
export nonlinear_inversion, fit_spectra, run_inversion
export process_all_files

# some useful funcs fom utils.jl
export assemble_state_vector!, OCO_spectra, OrderedDict

end # module
