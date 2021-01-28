using RadiativeTransfer.CrossSection
using HDF5, Statistics , Interpolations, Dates, RecursiveArrayTools, NCDatasets

const c = 299792458 * 100 # speed of light cm/s
const r = 8.314472; # universal gas constant joules / moles / K
const Nₐ = 6.0221415e23; # molecules / moles

struct MolecularMetaData
    datafile::String
    molecule_num::Integer
    isotope_num::Integer
    ν_range::Array{<:Real,1}
    hitran_table
end

mutable struct Molecule
    cross_sections
    grid
    p
    T
    hitran_table
end


mutable struct Spectra
    H₂O::Molecule
    CH₄::Molecule
    CO₂::Molecule
    HDO::Molecule
end


struct InversionResults
    timestamp
    x
    y
    f
    χ²
    S
    grid
end

function get_molecule_info(datafile::String, molecule_num::Int, isotope_num::Int, ν_range::Array{<:Real,1})
        hitran_table = CrossSection.read_hitran(datafile, mol=molecule_num, iso=isotope_num, ν_min=ν_range[1], ν_max=ν_range[2])
        return MolecularMetaData(datafile, molecule_num, isotope_num, ν_range, hitran_table)
    end


function calculate_cross_sections( filename::String, molec_num::Integer, iso_num::Integer; ν_min::Real=6000, ν_max::Real=6400, δν=0.01, p::Real=1001, T::Real=290)
    hitran_table = CrossSection.read_hitran(filename, mol=molec_num, iso=iso_num, ν_min=ν_min, ν_max=ν_max)
    model = make_hitran_model(hitran_table, Voigt());
    grid = ν_min:δν:ν_max;
    cross_sections = absorption_cross_section(model, grid, p, T);
    
    # store results in a struct
    molecule = Molecule(cross_sections, grid, p, T, hitran_table)
    return molecule
end #function calculate_cross_sections

function calculate_cross_sections!(molecule::Molecule; T::Real=290, p::Real=1001)
hitran_table = molecule.hitran_table;
    model = make_hitran_model(hitran_table, Voigt());
    grid = molecule.grid[1]:mean(diff(molecule.grid)):molecule.grid[end];
    
    value = absorption_cross_section(model, grid, p, T);
    
    # store rsults in a struct
    molecule = Molecule(value, grid, p, T, hitran_table)
    return molecule
end


function construct_spectra(H₂O_datafile::String, CH₄_datafile::String, CO₂_datafile::String, HDO_datafile::String; ν_min::Real=6000, ν_max::Real=6300, δν::Real=0.01, p::Real=1001, T::Real=290, use_TCCON=false)
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

function construct_spectra!(spectra::Spectra; p=1001, T=290)
    H₂O = calculate_cross_sections!(spectra.H₂O, p=p, T=T);
    CO₂ = calculate_cross_sections!(spectra.CO₂, p=p, T=T)
    CH₄ = calculate_cross_sections!(spectra.CH₄, p=p, T=T)
    HDO = calculate_cross_sections!(spectra.HDO, p=p, T=T)

    
    spectra.H₂O, spectra.CO₂, spectra.CH₄, spectra.HDO = H₂O, CO₂, CH₄, HDO;
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
    

function read_NIST_data(filename::String)
    data_fields = ["Temperature_K", "Pressure_mbar", "path_m", "LocalTime", "Freq_Hz", "DCSdata_Hz"];
    file = h5open(filename, "r")
    temperature = read(file, "Temperature_K") # temperature in K
    pressure = 0.9938*read(file, "Pressure_mbar") # pressure in mbar 
    pathlength = read(file, "path_m") .* 100; # path in cm
    time = read(file, "LocalTime")
    grid = read(file, "Freq_Hz")
    grid = grid ./ c # convert from hz to wavenumber 
    intensity = read(file, "DCSdata_Hz")
    close(file)

    # save to a struct
    dataset = FrequencyCombDataset(filename, intensity, grid, temperature, pressure, time, pathlength)
    return dataset
end # function read_NIST_data


function read_new_NIST_data(filename::String)
    data_fields = ["Temperature_K", "Pressure_mbar", "path_m", "LocalTime", "Freq_Hz", "DCSdata_Hz"];
    file = h5open(filename, "r")
    pressure = 0.9938*read(file, "pressure_evolution") # pressure in mbar
    temperature = read(file, "temperature_fitted_l") + 273.15*ones(length(pressure)) # temperature in K
    pathlength = 195017 # round trip path length in meters DCSA
    #pathlength = 196367 # round trip path length in m for DCSB

    time = read(file, "totaltime_priorfit")
    grid = read(file, "frequency_Hz")
    grid = grid ./ c # convert from hz to wavenumber 
    intensity = read(file, "WaveToFitROI_1600a")
    close(file)

    # save to a struct
    dataset = FrequencyCombDataset(filename, intensity, grid, temperature, pressure, time, pathlength)
    return dataset
end # function read_NIST_data


function take_time_average(dataset::FrequencyCombDataset; δt::Period=Dates.Hour(1))
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

        # update variablers
        i += 1
        t₁ = t₂
        t₂ = t₂ + δt
    end # while loop

    data_out = TimeAveragedFrequencyCombDataset(dataset.filename, averaged_measurements, dataset.grid, averaged_temperature, averaged_pressure, averaging_times, dataset.pathlength, num_averaged_measurements, δt, machine_time)
    return data_out
end

    



function find_indexes(ν_min::Real, ν_max::Real, ν_grid::Array{Float64,1})
    a = findlast(x -> x <= ν_min, ν_grid)
    b = findfirst(x -> x >= ν_max, ν_grid)
    indexes = collect(a:b)
    return indexes
end

function calc_vcd(p::Float64, T::Float64, δz::Float64, VMR_H₂O::Float64)
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end

function calc_vcd(p::Float64, T::Float64, δz::Float64)
    VMR_H₂O = 0
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end

    

function get_measurement(measurement_num::Integer, dataset::Dataset, ν_min::Real, ν_max::Real)
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
    itp = interpolate(σ, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, ν, T, p);
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
           defDim(file, "start_time", num_datapoints)
    defVar(file, "start_time", data.timestamp, ("start_time",))
    try
        for i=1:num_experiments
            println(experiment_label[i])
       timeseries = results[i,:]
           
       group = defGroup(file, experiment_label[i])
        defDim(group, "spectral_grid", length(timeseries[1].grid));
        defVar(group, "spectral_grid", timeseries[1].grid, ("spectral_grid",))
        model = defVar(group, "model", Float64, ("start_time", "spectral_grid"))
        measurement = defVar(group, "measurement", Float64, ("start_time", "spectral_grid"))

            # save to file
            println("saving measurement")
            measurement[:,:] = list2array([timeseries[i].y for i=1:num_datapoints])
            println("saving model")
        model[:,:] = list2array([timeseries[i].f for i=1:num_datapoints])
    end
    catch
        println("save failed")
    finally
        close(file)
    end
end
