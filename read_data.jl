using RadiativeTransfer, RadiativeTransfer.Absorption
using HDF5, Statistics , Interpolations, Dates, NCDatasets
using OrderedCollections

# define physical constants 
const c = 299792458 * 100 # speed of light cm/s
const r = 8.314472; # universal gas constant joules / moles / K
const Nₐ = 6.0221415e23; # molecules / moles
const g₀ = 9.8196 #gravity
global const H₂O_ind, CH₄_ind, CO₂_ind, HDO_ind = 1, 2, 3, 4;
global const temperature_ind, pressure_ind = 5, 6
global const windspeed_ind = 7


struct MolecularMetaData
    """
stores the metadata of hitran line-lists and coefficients in type HitranTable
"""
    filename::String
    molecule_num::Integer
    isotope_num::Integer
    ν_grid::AbstractRange
    hitran_table
    model
end


mutable struct Molecule
    """
type to store cross-sections, pressure, temperature, and line-list parameters
"""
    
    cross_sections
    grid
    p
    T
    hitran_table
end


mutable struct Spectra
    """
Type to store molecules
"""
    H₂O::Molecule
    CH₄::Molecule
    CO₂::Molecule
    HDO::Molecule
end


struct InversionResults
    """
-type for storing results from spectral fit
- used later for saving into NetCDF files
"""
    timestamp
    x
    measurement
    model
    χ²
    S
    grid
    K
    Sₑ
    Sₐ
end

function get_molecule_info(filename::String, molecule_num::Int, isotope_num::Int, ν_grid::AbstractRange)
    """
constructor for MolecularMetaData
"""
    hitran_table = read_hitran(filename, mol=molecule_num, iso=isotope_num, ν_min=ν_range[1], ν_max=ν_range[end])
    model = make_hitran_model(hitran_table, Voigt(), architecture=CPU());
        return MolecularMetaData(filename, molecule_num, isotope_num, ν_grid, hitran_table, model)
    end


function calculate_cross_sections( filename::String, molec_num::Integer, iso_num::Integer; ν_min::Real=6000, ν_max::Real=6400, δν=0.01, p::Real=1001, T::Real=290)
    """
- Constructor for Molecule type
- calculates the cross-sections of a HiTran molecule and stores in Molecule type
"""
    
    # retrieve the HiTran parameters 
    hitran_table = read_hitran(filename, mol=molec_num, iso=iso_num, ν_min=ν_min, ν_max=ν_max)
    model = make_hitran_model(hitran_table, Voigt(), architecture=CPU());
    grid = ν_min:δν:ν_max;
    cross_sections = absorption_cross_section(model, grid, p, T)
    
    # store results in the Molecule type
    molecule = Molecule(cross_sections, grid, p, T, hitran_table)
    return molecule
end #function calculate_cross_sections

function construct_spectra(molecules::Array{MolecularMetaData,1}; ν_min::Real=6000, δν::Real=0.01, ν_max::Real=6300, p::Real=1001, T::Real=295, use_TCCON::Bool=false)
    """
- Calculates the cross-sections of all input molecules inputted as type MolecularMetaData
- returns Molecules as a Dict
"""
    
    ν_grid = ν_min:δν:ν_max;
    cross_sections = map(x -> absorption_cross_section(x.model, ν_grid, p, T), molecules) # store results in a struct
    out = OrderedDict(molecules[i] => Molecule(cross_sections[i], collect(ν_grid), p, T, molecules[i].hitran_table) for i=1:length(molecules))
    return out
end #function calculate_cross_sections




function calculate_cross_sections!(molecule::Molecule; T::Real=290, p::Real=1001)
    """
- recalculates the cross-sections given the Molecule type
- used in the forward model for institue cross-sections calculation 
"""
        model = make_hitran_model(molecule.hitran_table, Voigt(), architecture=CPU());
    grid = molecule.grid[1]:mean(diff(molecule.grid)):molecule.grid[end];

    # recalculate cross-sections
    cross_sections = absorption_cross_section(model, grid, p, T);
    
    # store rsults in a struct
    molecule = Molecule(cross_sections, grid, p, T, molecule.hitran_table)
    return molecule
end


function construct_spectra(H₂O_datafile::String, CH₄_datafile::String, CO₂_datafile::String, HDO_datafile::String; ν_min::Real=6000, ν_max::Real=6300, δν::Real=0.01, p::Real=1001, T::Real=290, use_TCCON=false)
    """
- Constructor for Spectra type
- Calculates the cross-sections of H₂O, CH₄, CO₂, and HDO
"""
    
    H₂O = calculate_cross_sections(H₂O_datafile, 1, 1, ν_min=ν_min, ν_max=ν_max, δν=δν, p=p, T=T)
    if use_TCCON
        HDO = calculate_cross_sections(HDO_datafile, 49, 1, ν_min=ν_min, ν_max=ν_max, δν=δν, p=p, T=T)
        elseif use_TCCON == false
        HDO = calculate_cross_sections(HDO_datafile, 1, 4, ν_min=ν_min, ν_max=ν_max, δν=δν, p=p, T=T)
    end
    
    CH₄ = calculate_cross_sections(CH₄_datafile, 6, 1, ν_min=ν_min, ν_max=ν_max, δν=δν, p=p, T=T)        
    CO₂ = calculate_cross_sections(CO₂_datafile, 2, 1, ν_min=ν_min, ν_max=ν_max, δν=δν, p=p, T=T)
    spectra = Spectra(H₂O, CH₄, CO₂, HDO)
    return spectra
end

function construct_spectra(molecules::Array{MolecularMetaData,1})
    spectra = Dict{MolecularMetaData, Molecule}(molecule => calculate_cross_sections(molecule) for molecule in molecules)
    return spectra
end



function construct_spectra!(spectra::Spectra; p::Real=1001, T::Real=290)
    """
recalcualtes the cross-sections of Molecules type stored in the Spectra type
"""
    
    H₂O = calculate_cross_sections!(spectra.H₂O, p=p, T=T);
    CO₂ = calculate_cross_sections!(spectra.CO₂, p=p, T=T)
    CH₄ = calculate_cross_sections!(spectra.CH₄, p=p, T=T)
    HDO = calculate_cross_sections!(spectra.HDO, p=p, T=T)

    
    spectra.H₂O, spectra.CO₂, spectra.CH₄, spectra.HDO = H₂O, CO₂, CH₄, HDO;
    return spectra
end

function construct_spectra!(spectra::AbstractDict; p::Real=1001, T::Real=290)
    for species in keys(spectra)
        spectra[species] = calculate_cross_sections!(spectra[species], p=p, T=T);
    end
    return spectra
end


    


abstract type Dataset end

abstract type FrequencyComb <: Dataset end

mutable struct FrequencyCombDataset <: Dataset
    filename::String
    intensity::Array{Float64,2}
    grid::Vector{Float64}
    temperature::Vector{Float64}
    pressure::Vector{Float64}
    time::Vector{Any}
    pathlength::Float64
end

struct TimeAveragedFrequencyCombDataset <: FrequencyComb
    filename::String
    intensity::Array{Float64,2}
    grid::Vector{Float64}
    temperature::Vector{Float64}
    pressure::Vector{Float64}
    time::Vector{Tuple{DateTime,DateTime}}
    pathlength::Float64
    num_averaged_measurements::Vector{Int64}
    averaging_window::Dates.Period
    timestamp::Array{Float64,1}
end


abstract type Measurement end

mutable struct FrequencyCombMeasurement <: Measurement
    intensity::Array{Float64,1}
    grid::Vector{Float64}
    temperature::Float64
    pressure::Float64
    time::Any
    pathlength::Float64
    vcd::Float64
    num_averaged_measurements::Int64
    averaging_window::Any
end
    


function read_DCS_data(filename::String)
    """
Constructor of the FrequencyCombDataset type 
"""

    file = h5open(filename, "r")
    ds_obj = read(file)
    if haskey(ds_obj, "pressure_evolution") && haskey(ds_obj, "totaltime_priorfit")
        datafields = ["pressure_evolution", "temperature_fitted_l", "totaltime_priorfit", "frequency_Hz", "WaveToFitROI_1600a"];
    elseif haskey(ds_obj, "pressure_evolution") && haskey(ds_obj, "TotalTime")
        datafields = ["pressure_evolution", "temperature_fitted_l", "TotalTime", "frequency_Hz", "WaveToFitROI_1600a"];
        elseif haskey(ds_obj, "Pres_CS")
            datafields = ["Pres_CS", "Temp_CS", "TotalTime", "frequency_Hz", "WaveToFitROI_1600a"]
        else
            println("incorrect HDF datafile fieldnames")
        end
       

	if  haskey(ds_obj, "temperature_fitted")
datafields[2] = "temperature_fitted"
elseif haskey(ds_obj, "temperature_fitted_l")
datafields[2] = "temperature_fitted_l"
elseif haskey(ds_obj, "Temp_CS")
            datafields[2] = "Temp_CS"
        elseif haskey(ds_obj, "temperature_evolution")
            datafields[2] = "temperature_evolution"
end

    pressure = 0.9938*read(file, datafields[1]) # pressure in mbar
    temperature = read(file, datafields[2]) + 273.15*ones(length(pressure)) # temperature in K
    pathlength = 195017 # round trip path length in meters DCSA
    #pathlength = 196367 # round trip path length in m for DCSB

    time = read(file, datafields[3])
    grid = read(file, datafields[4])
    grid = grid ./ c # convert from hz to wavenumber 
    intensity = read(file, datafields[5])
    close(file)

    # save to a struct
    dataset = FrequencyCombDataset(filename, intensity, grid, temperature, pressure, time, pathlength)
    return dataset
end # function read_DCS_data


function take_time_average(dataset::FrequencyCombDataset; δt::Period=Dates.Hour(1))
    """
-Takes time-average of the FrequencyCombDataset over a time δt::TimeDelta
- returns a TimeAveragedFrequencyCombDataset 
"""
    
    timestamps = unix2datetime.(dataset.time)
    t₁ = floor(timestamps[1], Dates.Hour)
    t₂ = t₁ + δt
    t_final = timestamps[end]

    num_measurements = ceil((t_final - t₁), Dates.Hour)
    num_measurements = Int(num_measurements/δt)
    averaged_measurements = Array{Float64}(undef,(num_measurements, size(dataset.intensity)[2]))
    averaged_temperature = Array{Float64}(undef, num_measurements)
    averaged_pressure = Array{Float64}(undef, num_measurements)
        averaging_times = Array{Tuple{DateTime,DateTime}}(undef, num_measurements)
    num_averaged_measurements = Array{Int64}(undef, num_measurements)
    machine_time = Array{Float64}(undef,num_measurements)
    i= 1

    while t₁ < t_final
        indexes = findall(t->(t>=t₁ && t<=t₂), timestamps)

        averaged_measurements[i,:] = mean(dataset.intensity[indexes, :], dims=1)
        averaged_temperature[i] = mean(dataset.temperature[indexes])
        averaged_pressure[i] = mean(dataset.pressure[indexes])
        num_averaged_measurements[i] = length(indexes) # save number of averaged measurements per window
        averaging_times[i] = (t₁, t₂)
        machine_time[i] = dataset.time[indexes[1]];

        # update variables
        i += 1
        t₁ = t₂
        t₂ = t₂ + δt
    end # while loop

    data_out = TimeAveragedFrequencyCombDataset(dataset.filename, averaged_measurements, dataset.grid, averaged_temperature, averaged_pressure, averaging_times, dataset.pathlength, num_averaged_measurements, δt, machine_time)
    return data_out
end

    



function find_indexes(ν_min::Real, ν_max::Real, ν_grid::Array{Float64,1})
    """
Finds the indexes given values ν_min:ν_max
"""
    
    a = findlast(x -> x <= ν_min, ν_grid)
    b = findfirst(x -> x >= ν_max, ν_grid)
    indexes = collect(a:b)
    return indexes
end
   
function calc_vcd(p::Real, T::Real, δz::Real)
    VMR_H₂O = 0
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4 # molecules/cm³
    vcd = δz*ρₙ # molecules/cm²
    return vcd
end


function vcd_pressure(δp::Real, T::Real, vmr_H₂O::Real)
    δp = δp*100 # convert from mbar to pascals 
    dry_mass = 28.9647e-3  /Nₐ  # in kg/molec, weighted average for N2 and O2
    wet_mass = 18.01528e-3 /Nₐ  # just H2O
    ratio = dry_mass/wet_mass
    vmr_dry = 1 - vmr_H₂O
    M  = vmr_dry * dry_mass + vmr_H₂O * wet_mass
    vcd_dry = vmr_dry*δp/(M*g₀*100.0^2)   #includes m2->cm2
    vcd_H₂O = vmr_H₂O*δp/(M*g₀*100^2)
    return vcd_dry #+ vcd_H₂O
end

   
function get_measurement(measurement_num::Integer, dataset::Dataset, ν_min::Real, ν_max::Real)
    """
Subsets the FrequencyCombDataset into indivitual measurements 
Constructor of Measurement type
"""
    
    i = measurement_num
    p = dataset.pressure[i]
    T = dataset.temperature[i]
    δz = dataset.pathlength
    time = dataset.time[i]

    # find indexes
    indexes = find_indexes(ν_min, ν_max, dataset.grid)
    grid = dataset.grid[indexes]
    intensity = dataset.intensity[i,indexes]
    vcd = calc_vcd(p, T, δz)

    # save to struct FrequencyCombMeasurement
    if typeof(dataset) == FrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, 1, 0)
        elseif typeof(dataset) == TimeAveragedFrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, dataset.num_averaged_measurements[i], dataset.averaging_window)
    end # if statement
    
    return measurement
    end # function get_measurement
    
function OCO_spectra(filename::String)
    """
Constructs an interpolation of the OCO cross-sections grid provided by JPL
-returns sitp::function(ν, T, p)
"""
    
    fid = h5open(filename, "r")
    p = read(fid, "Pressure") * 0.01 # convert from Pa to mbar 
    T = read(fid, "Temperature")[:,1];
    ν = read(fid, "Wavenumber")[19002:end]
    σ = read(fid, "Gas_02_Absorption")[19002:end, 2,:,:]; # cross-sections
    broadener = read(fid, "Broadener_01_VMR")
    close(fid)
    
    ### start interpolation 
    # make grid tuples
    p_min, p_max = p[1], p[end];
    δp = mean(diff(p));
    T_min, T_max = T[1], T[end];
    δT= mean(diff(T));
    ν_min, ν_max = ν[1], ν[end];
    δν = mean(diff(ν))
    broadener_min, broadener_max = broadener[1], broadener[end];
    δ_broadener = mean(diff(broadener));
    
    # create the grid
    p = p_min:δp:p_max;
    T = T_min:δT:T_max;
    ν = ν_min:δν:ν_max;
    broadener = broadener_min:δ_broadener:broadener_max;
    grid = (ν, broadener, T, p);
#    sitp = interpolate(grid, σ, Gridded(Linear()))
    itp = interpolate( σ, BSpline(Cubic(Line(OnGrid()))))
    sitp = Interpolations.scale(itp, ν, T, p);
    return sitp
end



function list2array(array_in::Array)
    outer_dim, inner_dim = length(array_in), length(array_in[1])
    data_type = typeof(array_in[1][1])

    # allocate memory
    array_out = Array{data_type}(undef, (outer_dim, inner_dim))
    v = VectorOfArray(array_in)
    array_out = convert(Array, v)
    return array_out'
end

function list2array!(array_out::Array, array_in::Array)
    v = VectorOfArray(array_in)
    array_out = convert(Array, v)
    return array_out'
end


function save_inversion_results(filename::String, results::Array{InversionResults}, data::Dataset, experiment_label::Array{String})
    
    file = NCDataset(filename, "c");
    num_experiments, num_datapoints = size(results);

    # create the dimensions
    #num_datapoints = 2
           defDim(file, "start_time", num_datapoints)
    defVar(file, "start_time", data.timestamp, ("start_time",))

    # save the concentrations
    H₂O_VMR = defVar(file, "H2O_VMR", Float64, ("start_time",))
    H₂O_VMR[:] = [results[1,i].x[H₂O_ind] for i =1:num_datapoints]
    
    CO₂_VMR = defVar(file, "CO2_VMR", Float64, ("start_time",))
    CO₂_VMR[:] = [results[1,i].x[CO₂_ind] for i=1:num_datapoints]
    CH₄_VMR = defVar(file, "CH4_VMR", Float64, ("start_time",))
    CH₄_VMR[:] = [results[CH₄_ind,i].x[CH₄_ind] for i=1:num_datapoints];
    HDO_VMR = defVar(file, "HDO_VMR", Float64, ("start_time",))
    HDO_VMR[:] = [results[3,i].x[HDO_ind] for i=1:num_datapoints];

    pressure = defVar(file, "pressure", Float64, ("start_time",))
    pressure[:] = [results[1,i].x[pressure_ind] for i=1:num_datapoints];

    temperature = defVar(file, "temperature", Float64, ("start_time",))
    temperature[:] = [results[1,i].x[temperature_ind] for i=1:num_datapoints];
    try

    

        for i=1:num_experiments
            println(experiment_label[i])
       timeseries = results[i,:]
           
       group = defGroup(file, experiment_label[i])
        defDim(group, "spectral_grid", length(timeseries[1].grid))
        defVar(group, "spectral_grid", timeseries[1].grid, ("spectral_grid",))
        model = defVar(group, "model", Float64, ("start_time", "spectral_grid"))
        measurement = defVar(group, "measurement", Float64, ("start_time", "spectral_grid"))

            # save to file
            println("saving measurement")
            measurement[:,:] = list2array([timeseries[i].y for i=1:num_datapoints])
            println("saving model")
        model[:,:] = list2array([timeseries[i].f for i=1:num_datapoints])
    end
    catch e
        println("save failed")
        println(e)
    finally
        close(file)
    end
end

