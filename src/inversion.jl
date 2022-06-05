
function make_obs_error(measurement::AbstractMeasurement;
                        σ²::Union{Nothing, Float64}=nothing,
                        masked_indexes::Union{Vector{Int64}, Nothing}=nothing)
    
    n = length(measurement.intensity)
    base = mean(measurement.intensity)

    if σ²==nothing # get noise from the instrument
        noise = measurement.σ²
    else #get noise from user 
        noise = σ²
    end
    
    value = @. 1/noise * ones(n)
    Sₑ⁻¹ = Diagonal(value)

    if masked_indexes != nothing
        for i in masked_indexes
            Sₑ⁻¹[i,i] = 1/(1e5 * noise)
        end
    end
    
        
    return Sₑ⁻¹ 
end




"""Make prior error covarience matrix"""
function make_prior_error(σ::Union{Array{<:Real,1}, OrderedDict})
    if typeof(σ) <: AbstractDict
        σ = assemble_state_vector!(σ)
    end
    
    Sₐ⁻¹ = Diagonal((1 ./ σ).^2)
    return Sₐ⁻¹
end




function failed_inversion(xₐ::OrderedDict, measurement::AbstractMeasurement)

    # define an x vector of NaNs 
    x_error = OrderedDict(key => NaN*xₐ[key] for key in keys(xₐ))    
    return InversionResults(measurement.time, measurement.machine_time, x_error, measurement.intensity, NaN*measurement.intensity, NaN, NaN, measurement.grid, NaN, NaN, NaN)
end


function nonlinear_inversion(f, x₀::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)

    if haskey(inversion_setup, "obs_covariance")
        println("Using user-defined covariance")
        Sₑ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_indexes")
        println("masking out selected wave-numbers")
        Sₑ⁻¹ = make_obs_error(measurement, masked_indexes=inversion_setup["masked_indexes"])
    else
        println("default covariance")
        Sₑ⁻¹ = make_obs_error(measurement)
    end
    
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
    #cfg = ForwardDiff.JacobianConfig(f,xᵢ, ForwardDiff.Chunk{chunk_size}())
    result = DiffResults.JacobianResult(measurement.grid, xᵢ);
    jf! = (out, _x) -> ForwardDiff.jacobian!(out, f, _x)
    
    # begin the non-linear fit
     while i<10 && δᵢ>tolerence

        # evaluate the model and jacobian
        
        #result = DiffResults.JacobianResult(measurement.grid, xᵢ);
         #ForwardDiff.jacobian!(result, f, xᵢ)#,
         @time result = jf!(result, xᵢ)
         x_old = copy(xᵢ)
        fᵢ[:], kᵢ[:,:] = result.value, result.derivs[1]

        # Gauss-Newton Algorithm
         xᵢ[:] = xᵢ + inv(kᵢ'* Sₑ⁻¹ *kᵢ)*kᵢ'* Sₑ⁻¹ *(y - fᵢ);

        #evaluate relative difference between this and previous iteration 
         #δᵢ = abs((norm( fᵢ .- y) .- norm(f_old .- y)) ./ norm(f_old .- y));
         δᵢ = abs(norm( x_old .- xᵢ) ./ norm(x_old));
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
    χ² = (y-fᵢ)'* Sₑ⁻¹ *(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(kᵢ'*Sₑ⁻¹*kᵢ); # posterior error covarience

    x=assemble_state_vector!(xᵢ, collect(keys(x₀)), inversion_setup)
    return InversionResults(timestamp=measurement.time, machine_time=measurement.machine_time,
                              x=x,
                              measurement=y, model=fᵢ, χ²=χ², S=S,
                              grid=measurement.grid, K=kᵢ, Sₑ⁻¹=Sₑ⁻¹, Sₐ⁻¹=ones(length(measurement.grid))) 
    
end#function


"""fit over an atmospheric column with multiple layers"""
function profile_inversion(f::Function, x₀::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)

    # define the observational prior error covariance
        if haskey(inversion_setup, "obs_covariance")
        println("Using user-defined covariance")
        Sₒ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_indexes")
        println("masking out selected wave-numbers")
        Sₒ⁻¹ = make_obs_error(measurement, masked_indexes=inversion_setup["masked_indexes"])
    else
        println("default covariance")
        Sₒ⁻¹ = make_obs_error(measurement)
    end
    
    
    # state vectors 
    xₐ = assemble_state_vector!(x₀); # apriori
    Sₐ⁻¹ = make_prior_error(inversion_setup["σ"]); # a priori covarience  matrix 
    xᵢ = copy(xₐ); # current state vector 
    
    num_levels = length(measurement.pressure)
    
    tolerence = 1.0e-4; # relative error reached to stop loop
    γ = 1.0; # regularization parameter 
    δᵢ = 15.0; # relative errror 
    i = 1; # iteration count 

    # allocate memory for inversion matrixes
    y = measurement.intensity; # obserbations 
    Kᵢ = zeros((length(measurement.grid), length(xᵢ))) # jacobian
    f_old = similar(y) # previous model-run 
    fᵢ = similar(y) # current model-run

    # begin the non-linear fit
    while i<15 && δᵢ>tolerence

        # evaluate the model and jacobian 
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), xᵢ);
        ForwardDiff.jacobian!(result, f, xᵢ);
        fᵢ, Kᵢ = result.value, result.derivs[1]

        # Baysian Maximum Likelihood Estimation 
        lhs = (Sₐ⁻¹ + Kᵢ'*Sₒ⁻¹*Kᵢ + γ*Sₐ⁻¹)
        rhs = (Kᵢ'*Sₒ⁻¹ * (y - fᵢ) - Sₐ⁻¹*(xᵢ - xₐ))
        Δx = lhs\rhs
        xᵢ = xᵢ + Δx; # reassign state vector for next iteration

        #evaluate relative difference between this and previous iteration 
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));
        if inversion_setup["verbose_mode"]
            println("δᵢ for iteration ",i," is ",δᵢ)
        end
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
        end

        i = i+1
        f_old = fᵢ # reassign model output 
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(Kᵢ'*Sₒ⁻¹*Kᵢ); # posterior error covarience
    x=assemble_state_vector!(xᵢ, collect(keys(x₀)), num_levels, inversion_setup)

    # Gain matrix
    return InversionResults(timestamp=measurement.time, machine_time=measurement.machine_time,
                              x=x,
                              measurement=y, model=fᵢ, χ²=χ², S=S,
                              grid=measurement.grid, K=Kᵢ, Sₑ⁻¹=Sₒ⁻¹, Sₐ⁻¹=Sₐ⁻¹)
end#function    



function fit_spectra(measurement_num::Integer, xₐ::AbstractDict, dataset::AbstractDataset, spectra::AbstractDict, ν_range::Tuple, inversion_setup::Dict{String,Any})
    
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



function run_inversion(xₐ::AbstractDict, dataset::AbstractDataset, molecules::Array{MolecularMetaData,1}, inversion_setup::Dict, spectral_windows::AbstractDict)
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

function run_inversion(xₐ::AbstractDict, dataset::AbstractDataset, molecules::Array{MolecularMetaData,1}, inversion_setup::Dict, spectral_windows::Vector)
    
    num_measurements = length(dataset.pressure) # number of total measurements
    modelled = Array{InversionResults}(undef, num_measurements)
    num_windows = length(spectral_windows);
    results = Array{InversionResults}(undef, (num_windows, num_measurements));
    println("Beginning inversion")
    
    for (j, spectral_window) in enumerate(spectral_windows)

        spectra = construct_spectra(molecules, ν_grid=spectral_window[1]-0.1:0.01:spectral_window[end]+0.1, T=xₐ["temperature"], p=xₐ["pressure"])
        out = pmap(i -> fit_spectra(i, xₐ, dataset, spectra, spectral_window, inversion_setup), 1:num_measurements)
        results[j,:] = out;

    end
    return results
end



        


        
function process_all_files(xₐ::AbstractDict,
                           dataset::AbstractDataset,
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
                           dataset::AbstractDataset,
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
