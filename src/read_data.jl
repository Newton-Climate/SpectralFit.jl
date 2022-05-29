include("constants.jl")
include("types.jl")
include("utils.jl")

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
    pathlength = 195017.0 # round trip path length in meters DCSA
    #pathlength = 196367 # round trip path length in m for DCSB

    machine_time = HDF5.read(file, datafields[3])
    grid = HDF5.read(file, datafields[4])
    grid = grid ./ c # convert from hz to wavenumber 
    intensity = HDF5.read(file, datafields[5])
    close(file)

    # calculate noise
    σ² = calc_DCS_noise(grid, intensity)
    averaging_window = Second(1)
    n = length(pressure)
    num_averaged_measurements = ones(Int64, n)
    vcd = calc_vcd.(pressure, temperature, pathlength)
    time = unix2datetime.(machine_time)


    dataset = FrequencyCombDataset(filename=filename, intensity=intensity, grid=grid,
                                              temperature=temperature, pressure=pressure,
                                time=time, pathlength=pathlength,                                              num_averaged_measurements=num_averaged_measurements, averaging_window=averaging_window, machine_time=machine_time, σ²=σ², vcd=vcd)
    
    # save to a struct
    #dataset = map(i->FrequencyCombMeasurement(intensity=intensity[i,:], grid=grid,
#                                              temperature=temperature[i], pressure=pressure[i],
#                                              time=time[i], #pathlength=pathlength,                                              #num_averaged_measurements=num_averaged_measurements, averaging_window=averaging_window, #machine_time=machine_time[i], σ²=σ²[i], vcd=vcd[i]), 1:n)
    return dataset
end # function read_DCS_data





function take_time_average!(dataset::FrequencyCombDataset; δt::Period=Dates.Hour(1))
    """
-Takes time-average of the FrequencyCombDataset over a time δt::TimeDelta
- returns a TimeAveragedFrequencyCombDataset 
"""

    FT = eltype(dataset.intensity)
    IT = eltype(dataset.num_averaged_measurements)
    timestamps = dataset.time
    t₁ = floor(timestamps[1], δt)
    t₂ = t₁ + δt
    t_final = timestamps[end]
    num_measurements = ceil((t_final - t₁), δt)
    num_measurements = Int(num_measurements/δt)

    # allocate memory 
    averaged_measurements = Array{FT}(undef,(num_measurements, size(dataset.intensity)[2]))
    averaged_temperature = Array{FT}(undef, num_measurements)
    averaged_pressure = Array{FT}(undef, num_measurements)
    averaging_times = Array{DateTime}(undef, num_measurements)
    averaged_σ² = Array{FT,1}(undef, num_measurements)
    num_averaged_measurements = Array{IT}(undef, num_measurements)
    machine_time = Array{FT}(undef,num_measurements)
    averaged_vcd = similar(averaged_pressure)
    i= 1;

    while t₁ < t_final
        indexes = findall(t->(t>=t₁ && t<=t₂), timestamps)
        
        # in the case where we get no idnexes, this averaging won't fail
        length(indexes)<1 ? n=length(indexes) : break

        averaged_measurements[i,:] = mean(dataset.intensity[indexes, :], dims=1)
        averaged_temperature[i] = mean(dataset.temperature[indexes])
        averaged_pressure[i] = mean(dataset.pressure[indexes])
        averaged_σ²[i] = 1/n * mean(dataset.σ²[indexes])
        num_averaged_measurements[i] = n
        averaging_times[i] = t₁
        machine_time[i] = dataset.time[indexes[1]]
        averaged_vcd[i] = mean(dataset.vcd[indexes])
        
        # update variables
        i += 1
        t₁ = t₂
        t₂ = t₂ + δt
    end # while loop

    dataset.intensity= averaged_measurements
                                    dataset.temperature=averaged_temperature
    dataset.pressure=averaged_pressure
    dataset.time=averaging_times
    dataset.num_averaged_measurements=num_averaged_measurements
    dataset.averaging_window=δt
    dataset.machine_time=machine_time
    dataset.σ²=averaged_σ²
    dataset.vcd=averaged_vcd
    return dataset
end



    
"""
Subsets the FrequencyCombDataset into indivitual measurements 
Constructor of Measurement type
"""
function get_measurement(measurement_num::Integer, dataset::AbstractDataset, ν_min::Real, ν_max::Real)

    i = measurement_num
    p = dataset.pressure[i]
    T = dataset.temperature[i]
    δz = dataset.pathlength
    time = dataset.time[i]

    # find indexes
    indexes = find_indexes(ν_min, ν_max, dataset.grid)
    grid = dataset.grid[indexes]
    intensity = dataset.intensity[i,indexes]

    # save to struct FrequencyCombMeasurement
    measurement = FrequencyCombMeasurement(intensity=intensity, grid=grid,
                                           temperature=T, pressure=p, time=time,
                                           pathlength=δz, vcd=dataset.vcd[i], num_averaged_measurements=dataset.num_averaged_measurements[i],
                                           averaging_window=dataset.averaging_window, machine_time=dataset.machine_time[i], σ²=dataset.σ²[i])
    
    return measurement
end # function get_measurement

function data2measurements(dataset::AbstractDataset, ν_min=nothing, ν_max=nothing)
    min = ν_min==nothing ? dataset.grid[1] : ν_min
    max = ν_max==nothing ? dataset.grid[end] : ν_max
    
    n = length(dataset.pressure)
    out = map(i->get_measurement(i, dataset, min, max), 1:n)
    return out
end

    

function get_measurement(measurement_num::Integer, dataset::AbstractDataset, ν_min::Real, ν_max::Real, p::Array{Float64,1}, T::Array{Float64,1})

    i = measurement_num
    δz = dataset.pathlength
    time = dataset.time[i]

    # find indexes
    indexes = find_indexes(ν_min, ν_max, dataset.grid)
    grid = dataset.grid[indexes]
    intensity = dataset.intensity[i,indexes]
    vcd = make_vcd_profile(p, T)

    # save to struct FrequencyCombMeasurement
        measurement = FrequencyCombMeasurement(intensity=intensity, grid=grid,
                                           temperature=T, pressure=p, time=time,
                                           pathlength=δz, vcd=vcd, num_averaged_measurements=dataset.num_averaged_measurements[i],
                                           averaging_window=dataset.averaging_window, machine_time=dataset.machine_time[i], σ²=dataset.σ²[i])
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

data = read_DCS_data("../20160921.h5")
data = take_time_average!(data)
measurement = get_measurement(1, data, 6050, 6120)
