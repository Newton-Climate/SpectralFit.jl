using JLD2, vSmartMOM, OrderedCollections, vSmartMOM.Absorption
include("types.jl")

function load_interp_model(filepath::String)
    @load filepath itp_model
    return itp_model
end


"""constructor for MolecularMetaData
Stores parameters for the HiTran parameters
Construct one object per molecule being analyzed"""
function get_molecule_info(molecule::String, filename::String, molecule_num::Int, isotope_num::Int, ν_grid::AbstractRange{<:Real}; architecture=CPU())

    hitran_table = read_hitran(filename, mol=molecule_num, iso=isotope_num, ν_min=ν_grid[1], ν_max=ν_grid[end])
    model = make_hitran_model(hitran_table, Voigt(), architecture=architecture);
    return MolecularMetaData(molecule=molecule, filename=filename,
                             molecule_num=molecule_num, isotope_num=isotope_num,
                             ν_grid=ν_grid, model=model)
end

function get_molecule_info(molecule::String, filepath::String; hitran_table=nothing)
    model = load_interpolation_model(filepath)
    #hitran_table = read_hitran(filepath, mol=model.mol, iso=model.iso, ν_min=model.ν_grid[1], ν_max=model.ν_grid[end])
        return MolecularMetaData(molecule, filepath, model.mol, model.iso, model.ν_grid, itp_model)
    end


"""
- Constructor for Molecule type
- calculates the cross-sections of a HiTran molecule and stores in Molecule type
"""
function calculate_cross_sections( filename::String, molec_num::Integer, iso_num::Integer; ν_min::Real=6000, ν_max::Real=6400, δν=0.01, p::Real=1001, T::Real=290, architecture=CPU())

    # retrieve the HiTran parameters 
    hitran_table = CrossSection.read_hitran(filename, mol=molec_num, iso=iso_num, ν_min=ν_min, ν_max=ν_max)
    model = make_hitran_model(hitran_table, Voigt(), architecture=architecture);
    grid = ν_min:δν:ν_max;
    cross_sections::Array{Float64,1} = absorption_cross_section(model, grid, p, T)
    
    # store results in the Molecule type
    molecule = Molecule(cross_sections, grid, p, T, model)
    return molecule
end #function calculate_cross_sections

"""
- Calculates the cross-sections of all input molecules inputted as type MolecularMetaData
- returns Molecules as a Dict
"""
function construct_spectra(molecules::Array{MolecularMetaData,1}; ν_grid::AbstractRange{<:Real}=6000:0.1:6400, p::Real=1001, T::Real=295)
    
    cross_sections = map(x -> absorption_cross_section(x.model, ν_grid, p, T), molecules) # store results in a struct
    out = OrderedDict(molecules[i].molecule => Molecule(cross_sections=cross_sections[i], grid=ν_grid, p=p, T=T, model=molecules[i].model) for i=1:length(molecules))
    return out
end #function calculate_cross_sections




"""
- recalculates the cross-sections given the Molecule type
- used in the forward model for institue cross-sections calculation 
"""
function calculate_cross_sections!(molecule::Molecule; T::Real=290, p::Real=1001)
   
    # recalculate cross-sections
    molecule.cross_sections[:] = absorption_cross_section(molecule.model, molecule.grid, p, T);
    return molecule
end #function calculate_cross_sections!




function construct_spectra!(spectra::AbstractDict; p::Real=1001, T::Real=290)

    # iterate over species (key) and molecule(referenced object)
    for (species, molecule) in spectra
        spectra[species].cross_sections = absorption_cross_section(molecule.model, molecule.grid, p, T);
    end    
end


### define some default values for p and T
p_default = collect(range(450,800, length=20))
T_default = collect(range(240, 290, length=20))

"""calculate corss-sections for vertical profiles, where pressure and temperature are Arrays"""
function construct_spectra_by_layer(molecules::Array{MolecularMetaData,1}; p::Array{<:Real,1}=p_default, T::Array{<:Real,1}=T_default, ν_min=6000, ν_max=6400, δν=0.01)
    n_levels = length(p) 
    spectra = Array{OrderedDict}(undef, n_levels)
    for i=1:length(p)
        spectra[i] = construct_spectra(molecules, p=p[i], T=T[i], ν_grid=ν_min:δν:ν_max)
    end
    return spectra
end


function construct_spectra_by_layer!(spectra::Array{OrderedDict,1}; p::Array{<:Real,1}=p_default, T::Array{<:Real,1}=T_default)
    num_levels = length(p)
    for i=1:num_levels
        spectra[i] = construct_spectra!(spectra[i], p=p[i], T=T[i])
    end
    return spectra
end


inversion_setup = Dict{String,Any}(
    "poly_degree" => 100,
    "fit_pressure" => true,
    "fit_temperature" => true,
    "use_OCO" => false,
"use_TCCON" => false,
"verbose_mode" => true,
"architecture" => CPU(),
"fit_column" => false)

# Just defining the spectral windows for each species
ν_CH4 = (6050, 6120)
ν_range = ν_CH4[1]:ν_CH4[2]
ν_min , ν_max = ν_CH4[1]-1, ν_CH4[end]+1


CH₄ = get_molecule_info("CH4", "../data/hit20_12CH4.par", 6, 1, ν_range, architecture=inversion_setup["architecture"])
H₂O = get_molecule_info("H2O", "../data/hit20_H2O.par", 1, 1, ν_range, architecture=inversion_setup["architecture"])
CO₂ = get_molecule_info("CO2", "../data/hit20_12CO2.par", 2,1,ν_range, architecture=inversion_setup["architecture"])

# Calculate the cross-sections and store in dictionary
molecules = [H₂O, CH₄, CO₂]
spec = construct_spectra(molecules, ν_grid=ν_min:0.003:ν_max, p=1e3, T=290.0)
