using LinearAlgebra, Distributed, DistributedArrays, ForwardDiff, DiffResults
using ProgressMeter, JLD2




function make_obs_error(measurement::Measurement; σ=nothing)
    n = length(measurement.intensity)
    base = mean(measurement.intensity)

    if σ==nothing
        noise = measurement.σ / sqrt(measurement.num_averaged_measurements)
    else
        noise = σ
    end
    
    value = @. 1/noise^2 * ones(n)
    Sₑ = Diagonal(value)
    return Sₑ
end

function make_obs_error(dataset::Dataset; a::Float64=0.3*0.01611750368314077)
    n = length(measurement.intensity)
    base = mean(dataset.intensity)
        value = 1/(a*sqrt(base))^2 * ones(n)
    Sₑ = Diagonal(value)
    return Sₑ
end


"""Make prior error covarience matrix"""
function make_prior_error(σ::Union{Array{<:Real,1}, OrderedDict})
    if typeof(σ) <: AbstractDict
        σ = assemble_state_vector!(σ)
    end
    
    Sₐ⁻¹ = Diagonal((1 ./ σ).^2)
    return Sₐ⁻¹
end


function nonlinear_inversion(x₀::Array{<:Real,1}, measurement::Measurement, spectra::Spectra, inversion_setup::AbstractDict)
    f = generate_forward_model(measurement, spectra, inversion_setup);
    Sₑ = make_obs_error(measurement);
    y = measurement.intensity;
    kᵢ = zeros(length(y), length(x₀));
    xᵢ = x₀;
    tolerence = 1.0e-4;
    δᵢ = 10;
    i = 1
    fᵢ = f(xᵢ);


    while i<10 && δᵢ>tolerence
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), x₀);
        ForwardDiff.jacobian!(result, f, xᵢ);
        f_old = fᵢ

        fᵢ, kᵢ = result.value, result.derivs[1]


        x = xᵢ+inv(kᵢ'*Sₑ*kᵢ)*kᵢ'*Sₑ*(y-fᵢ);
        #x = xᵢ+inv(kᵢ'*kᵢ)*kᵢ'*(y-fᵢ);


        xᵢ = x;
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));
        
        if i==1 #prevent premature ending of while loop
            δᵢ = 1
        end
        if inversion_setup["verbose_mode"]
            println("δᵢ for iteration ",i," is ",δᵢ)
        end
        
        i = i+1
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₑ*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(kᵢ'*Sₑ*kᵢ)
    return InversionResults(measurement.time, measurement.machine_time, xᵢ, y, fᵢ, χ², S, measurement.grid, Kᵢ, Sₒ⁻¹, I)
end#function

function failed_inversion(xₐ::OrderedDict, measurement::Measurement)

    # define an x vector of NaNs 
    x_error = OrderedDict(key => NaN*xₐ[key] for key in keys(xₐ))    
    return InversionResults(measurement.time, measurement.machine_time, x_error, measurement.intensity, NaN*measurement.intensity, NaN, NaN, measurement.grid, NaN, NaN, NaN)
end


function nonlinear_inversion(f, x₀::AbstractDict, measurement::Measurement, spectra::AbstractDict, inversion_setup::AbstractDict)


    Sₑ = make_obs_error(measurement)
    #Sₑ = diagm(ones(length(measurement.intensity)));
    y = measurement.intensity;
    xᵢ = x₀;
    xᵢ = assemble_state_vector!(xᵢ)
    tolerence = 1.0e-4;
    δᵢ = 10.0;
    i = 1
    state_length, grid_length = length(xᵢ), length(measurement.grid)
        kᵢ = zeros(grid_length, state_length)
        fᵢ = zeros(grid_length)
        f_old = similar(fᵢ)
        chunk_size = state_length < 30 ? state_length : 20
    cfg = ForwardDiff.JacobianConfig(f,xᵢ, ForwardDiff.Chunk{chunk_size}())
    result = DiffResults.JacobianResult(measurement.grid, xᵢ);
    jf! = (out, _x) -> ForwardDiff.jacobian!(out, f, _x, cfg)
    
    # begin the non-linear fit
     while i<10 && δᵢ>tolerence

        # evaluate the model and jacobian
        
        #result = DiffResults.JacobianResult(measurement.grid, xᵢ);
         #ForwardDiff.jacobian!(result, f, xᵢ)#,
         result = jf!(result, xᵢ)
        fᵢ[:], kᵢ[:,:] = result.value, result.derivs[1]

        # Gauss-Newton Algorithm
         xᵢ[:] = xᵢ .+ inv(kᵢ'*Sₑ*kᵢ)*kᵢ'*Sₑ*(y .- fᵢ);
         

        #evaluate relative difference between this and previous iteration 
        δᵢ = abs((norm( fᵢ .- y) .- norm(f_old .- y)) ./ norm(f_old .- y));
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
        end

        if inversion_setup["verbose_mode"]
            println("δᵢ for iteration ",i," is ",δᵢ)
        end
        
         i = i+1
         f_old[:] = fᵢ
     end #while loop
    

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₑ*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(kᵢ'*Sₑ*kᵢ); # posterior error covarience

        output = InversionResults(measurement.time, measurement.machine_time, assemble_state_vector!(xᵢ, collect(keys(x₀)), inversion_setup), y, fᵢ, χ², S, measurement.grid, kᵢ, Sₑ, I)

    
    return output
end#function


"""fit over an atmospheric column with multiple layers"""
function nonlinear_inversion(f::Function, x₀::AbstractDict, measurement::Measurement, spectra::Array{<:AbstractDict,1}, inversion_setup::AbstractDict)
    

    Sₒ⁻¹ = SpectralFits.make_obs_error(measurement)
    
    #Sₑ = diagm(ones(length(measurement.intensity)));
    y = measurement.intensity;
    Kᵢ = zeros(length(y), length(x₀));
    xₐ = assemble_state_vector!(x₀);
    xᵢ = xₐ
    Sₐ⁻¹ = make_prior_error(inversion_setup["σ"])
    tolerence = 1.0e-4;
    γ = 1.0;
    δᵢ = 15;
    i = 1
    fᵢ = f(xᵢ)
    

    # begin the non-linear fit
    while i<15 && δᵢ>tolerence

        # evaluate the model and jacobian 
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), xᵢ);
        @time ForwardDiff.jacobian!(result, f, xᵢ);
        f_old = fᵢ # reassign model output 
        fᵢ, Kᵢ = result.value, result.derivs[1]

        # Gauss-Newton Algorithm
        lhs = (Sₐ⁻¹ + Kᵢ'*Sₒ⁻¹*Kᵢ + γ*Sₐ⁻¹)
        rhs = (Kᵢ'*Sₒ⁻¹ * (y - fᵢ) - Sₐ⁻¹*(xᵢ - xₐ))
        Δx = lhs\rhs

        x = xᵢ + Δx; # reassign state vector for next iteration
        xᵢ = x

        #evaluate relative difference between this and previous iteration 
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));        
        if i==1 #prevent premature ending of while loop
            δᵢ = 1
        end

        if inversion_setup["verbose_mode"]
            println("δᵢ for iteration ",i," is ",δᵢ)
        end
        
        i = i+1
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(Kᵢ'*Sₒ⁻¹*Kᵢ); # posterior error covarience

    # Gain matrix
    return InversionResults(measurement.time, measurement.machine_time, xᵢ, y, fᵢ, χ², S, measurement.grid, Kᵢ, Sₒ⁻¹, Sₐ⁻¹)
end#function    

function fit_spectra(measurement_num::Integer, xₐ::Array{<:Real,1}, dataset::Dataset, ν_range::Tuple)
    measurement = get_measurement(measurement_num, dataset, ν_range[1], ν_range[2])
    spectra = construct_spectra("../H2O_S.data", "../CH4_S.data", "../CO2_S.data", "../HDO_S.data", ν_min=ν_range[1]-3, ν_max=ν_range[2]+3, p=measurement.pressure, T=measurement.temperature, use_TCCON=inversion_setup["use_TCCON"])
    results = try
        nonlinear_inversion(xₐ, measurement, spectra, inversion_setup)
    catch
        failed_inversion(xₐ, measurement)
    end    
    return results
end


function fit_spectra(measurement_num::Integer, xₐ::AbstractDict, dataset::Dataset, spectra::AbstractDict, ν_range::Tuple, inversion_setup::Dict{String,Any})
    
    println(measurement_num)
    measurement = get_measurement(measurement_num, dataset, ν_range[1], ν_range[end])

    # Use a-priori pressure and temperature if given
    # Otherwise, use the measurement information 
        if haskey(xₐ, "pressure")
        p = xₐ["pressure"]
    else
        p = measurement.pressure
    end

    if haskey(xₐ, "temperature")
        T = xₐ["temperature"]
    else
        T = measurement.temperature
    end

#    spectra = construct_spectra(molecules, ν_grid=ν_range[1]-0.1:0.003:ν_range[end]+0.1, p=p, T=T, architecture=inversion_setup["architecture"])
    f = generate_forward_model(xₐ, measurement, spectra, inversion_setup)
    results = try
        nonlinear_inversion(f, xₐ, measurement, spectra, inversion_setup)
    catch
        println("Inversion for measurement ", measurement_num, " has failed.")
        failed_inversion(xₐ, measurement)
    end    
    return results
end



function run_inversion(xₐ::Array{<:Real,1}, dataset::Dataset, inversion_setup::Dict{String,Real})
    num_measurements = length(dataset.pressure) # number of total measurements
    modelled = Array{InversionResults}(undef, num_measurements)
    
    ν_CH₄ = (6055, 6120);
#ν_CO2 = (6206, 6280);
ν_CO₂ = (6180, 6250);
    ν_HDO = (6310,6380);

    results = Array{InversionResults}(undef, (3,num_measurements));
    println("Beginning inversion")
    
        Threads.@threads for i=1:num_measurements
  
        println(i)
        results[1,i] = fit_spectra(i, xₐ, dataset, ν_CO₂);        
        results[2,i] = fit_spectra(i, xₐ, dataset, ν_CH₄);
        results[3,i] = fit_spectra(i, xₐ, dataset, ν_HDO);
    end
    return results
end


function run_inversion(xₐ::AbstractDict, dataset::Dataset, molecules::Array{MolecularMetaData,1}, inversion_setup::Dict, spectral_windows::AbstractDict)
    num_measurements = length(dataset.pressure) # number of total measurements
    modelled = Array{InversionResults}(undef, num_measurements)
    num_windows = length(keys(spectral_windows));
    results = Array{InversionResults}(undef, (num_windows, num_measurements));
    println("Beginning inversion")
    
    for (j, spectral_window) in enumerate(keys(spectral_windows))

        spectra = construct_spectra(molecules, ν_grid=spectral_window[1]-0.1:0.01:spectral_window[end]+0.1, T=xₐ["temperature"], p=xₐ["pressure"])
        out = pmap(i -> fit_spectra(i, xₐ, dataset, spectra, spectral_window, inversion_setup), 1:num_measurements)
        results[j,:] = out;

    end
    return results
end



        


        
function process_all_files(xₐ::AbstractDict,
                           dataset::Dataset,
                           molecules::Array{MolecularMetaData,1},
                           inversion_setup::Dict,
                           spectral_windows::AbstractDict,
                           experiment_labels::Union{String, Array{String,1}};
                           data_path::String=pwd(),
                           out_path::String=pwd())

    if typeof(experiment_labels) <: Array{String,1}
        @assert length(spectral_windows) == length(experiment_labels)
    end
    
    files = readdir(data_path);
    num_files = length(files)
    
    for i=1:num_files
        
        file = files[i]
        full_file = data_path*file
        if endswith(file, ".h5") == false; continue; end;
        println(i,"/",num_files);
        println(file)
        
        data = read_DCS_data(full_file)
        data = take_time_average(data, δt=inversion_setup["averaging_window"])
        results = run_inversion(xₐ, data, molecules, inversion_setup, spectral_windows)

        outfile = out_path*"/"*file[1:end-3]*"_results.JLD2";
        #save_results(outfile, results, experiment_labels)
        @save outfile results
    end
    println("done with all files")
    return true
end


function process_all_files(xₐ::AbstractDict,
                           dataset::Dataset,
                           molecules::Array{MolecularMetaData,1},
                           inversion_setup::Dict,
                           spectral_windows::AbstractDict,
                           experiment_labels::Union{String, Array{String,1}},
                           datafiles::Array{String,1};
                           data_path::String=pwd(),
                           out_path::String=pwd())

    if typeof(experiment_labels) <: Array{String,1}
        @assert length(spectral_windows) == length(experiment_labels)
    end

    num_files = length(datafiles)

    for i=1:num_files
        file = data_path * datafiles[i]
        if endswith(file, ".h5") == false; continue; end;
        println(i,"/",num_files);
        println(file)
        
        data = read_DCS_data(file)
        data = take_time_average(data, δt=inversion_setup["averaging_window"])
        results = run_inversion(xₐ, data, molecules, inversion_setup, spectral_windows)

        outfile = out_path*"/"*datafiles[i][1:end-3]*"_results.JLD2";
        #save_results(outfile, results, experiment_labels)
        @save outfile results
    end
    println("done with all files")
    return true
end
