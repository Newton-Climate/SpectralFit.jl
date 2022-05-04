using RadiativeTransfer, RadiativeTransfer.Absorption
using Statistics , Interpolations, Dates
using OrderedCollections, HDF5, JLD2
    




"""Constructor of the FrequencyCombDataset type """
function read_DCS_data(filename::String)

    file = HDF5.h5open(filename, "r")
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

    pressure = 0.9938*HDF5.read(file, datafields[1]) # pressure in mbar
    temperature = HDF5.read(file, datafields[2]) + 273.15*ones(length(pressure)) # temperature in K
    pathlength = 195017 # round trip path length in meters DCSA
    #pathlength = 196367 # round trip path length in m for DCSB

    time = HDF5.read(file, datafields[3])
    grid = HDF5.read(file, datafields[4])
    grid = grid ./ c # convert from hz to wavenumber 
    intensity = HDF5.read(file, datafields[5])
    close(file)

    # calculate noise
    σ² = calc_DCS_noise(grid, intensity)
    num_averaged_measurements, averaging_window = 1, nothing

    # save to a struct
    dataset = map(i->FrequencyCombMeasurement(filename=filename, intensity=intensity[i,:], grid, temperature=temperature[i], pressure=pressure[i], time=time[i], pathlength=pathlength,                                              num_averaged_measurements=num_averaged_measurements[i], averaging_window=averaging_window[i], machine_time=time[i], σ²=σ²[i])
    return dataset
end # function read_DCS_data


function take_time_average(dataset::FrequencyCombDataset; δt::Period=Dates.Hour(1))
    """
-Takes time-average of the FrequencyCombDataset over a time δt::TimeDelta
- returns a TimeAveragedFrequencyCombDataset 
"""
    
    timestamps = unix2datetime.(dataset.time)
    t₁ = floor(timestamps[1], δt)
    t₂ = t₁ + δt
    t_final = timestamps[end]
    num_measurements = ceil((t_final - t₁), δt)
    num_measurements = Int(num_measurements/δt)
    averaged_measurements = Array{Float64}(undef,(num_measurements, size(dataset.intensity)[2]))
    averaged_temperature = Array{Float64}(undef, num_measurements)
    averaged_pressure = Array{Float64}(undef, num_measurements)
    averaging_times = Array{Tuple{DateTime,DateTime}}(undef, num_measurements)
    averaged_σ² = Array{Float64,1}(undef, num_measurements)
    num_averaged_measurements = Array{Int64}(undef, num_measurements)
    machine_time = Array{Float64}(undef,num_measurements)
    i= 1;

    while t₁ < t_final
        indexes = findall(t->(t>=t₁ && t<=t₂), timestamps)
        
        # in the case where we get no idnexes, this averaging won't fail
        if length(indexes) < 1; break; end;

        averaged_measurements[i,:] = mean(dataset.intensity[indexes, :], dims=1)
        averaged_temperature[i] = mean(dataset.temperature[indexes])
        averaged_pressure[i] = mean(dataset.pressure[indexes])
        averaged_σ²[i] = 1/length(indexes) * mean(dataset.σ²[indexes])
        num_averaged_measurements[i] = length(indexes) # save number of averaged measurements per window
        averaging_times[i] = (t₁, t₂)
        machine_time[i] = dataset.time[indexes[1]]
        

        # update variables
        i += 1
        t₁ = t₂
        t₂ = t₂ + δt
    end # while loop

    data_out = TimeAveragedFrequencyCombDataset(dataset.filename, averaged_measurements, dataset.grid, averaged_temperature, averaged_pressure, averaging_times, dataset.pathlength, num_averaged_measurements, δt, machine_time, averaged_σ²)
    return data_out
end

    






    
"""
Subsets the FrequencyCombDataset into indivitual measurements 
Constructor of Measurement type
"""
function get_measurement(measurement_num::Integer, dataset::Dataset, ν_min::Real, ν_max::Real)

    i = measurement_num
    p = dataset.pressure[i]
    T = dataset.temperature[i]
    δz = dataset.pathlength
    time = dataset.time[i]
    σ² = dataset.σ²[i]

    # find indexes
    indexes = find_indexes(ν_min, ν_max, dataset.grid)
    grid = dataset.grid[indexes]
    intensity = dataset.intensity[i,indexes]
    vcd = calc_vcd(p, T, δz)

    # save to struct FrequencyCombMeasurement
    if typeof(dataset) == FrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, 1, 0, dataset.timestamp[i], σ)
        elseif typeof(dataset) == TimeAveragedFrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, dataset.num_averaged_measurements[i], dataset.averaging_window, dataset.timestamp[i], σ²)
    end # if statement
    
    return measurement
end # function get_measurement


unction get_measurement(measurement_num::Integer, dataset::Dataset, ν_min::Real, ν_max::Real, p::Array{Float64,1}, T::Array{Float64,1})

    i = measurement_num
    δz = dataset.pathlength
    time = dataset.time[i]

    # find indexes
    indexes = find_indexes(ν_min, ν_max, dataset.grid)
    grid = dataset.grid[indexes]
    intensity = dataset.intensity[i,indexes]
    vcd = make_vcd_profile(p, T)

    # save to struct FrequencyCombMeasurement
    if typeof(dataset) == FrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, 1, 0, dataset.timestamp[i], dataset.σ[i])
        elseif typeof(dataset) == TimeAveragedFrequencyCombDataset
        measurement = FrequencyCombMeasurement(intensity, grid, T, p, time, δz, vcd, dataset.num_averaged_measurements[i], dataset.averaging_window, dataset.timestamp[i], dataset.σ[i])
    end # if statement
    
    return measurement
end # function get_measurement


    
"""
Constructs an interpolation of the OCO cross-sections grid provided by JPL
-returns sitp::function(ν, T, p)
"""
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


