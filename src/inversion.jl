
function make_obs_error(measurement::AbstractMeasurement;
                        σ²::Union{Nothing, Float64}=nothing,
                        masked_windows::Union{AbstractArray{<:Real}, Nothing}=nothing)
    
    n = length(measurement.intensity)
    base = mean(measurement.intensity)

    if σ²==nothing # get noise from the instrument
        noise = measurement.σ²
    else #get noise from user 
        noise = σ²
    end
    
    value = @. 1/noise * ones(n)
    Sₑ⁻¹ = Diagonal(value)

    if masked_windows != nothing
        masked_indexes = find_mask(measurement.grid, masked_windows)
        
        # the windows are outside the range of the grid
        if isempty(masked_indexes); return Sₑ⁻¹; end
        
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

function failed_vector(xₐ::AbstractDict)
    x_error = copy(xₐ)
    for key in keys(xₐ)
        x_error[key] = NaN*xₐ[key]
    end
    return x_error
end



function failed_inversion(xₐ::OrderedDict, measurement::AbstractMeasurement)

    # define an x vector of NaNs 
    x_error = failed_vector(xₐ)
    return FailedInversion(timestamp=measurement.time, machine_time=measurement.machine_time,
                              x=x_error,
                            measurement=measurement.intensity,
                            model = NaN*ones(length(measurement.intensity)),
                            grid=measurement.grid, χ²=NaN)
end


function nonlinear_inversion(f, x₀::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)

    verbose = inversion_setup["verbose_mode"]
    if haskey(inversion_setup, "obs_covariance")
        if verbose; println("Using user-defined covariance"); end
        Sₑ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_windows")
        if verbose; println("masking out selected wave-numbers"); end
        Sₑ⁻¹ = make_obs_error(measurement, masked_windows=inversion_setup["masked_windows"])
    else
        if verbose; println("default covariance"); end
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
         try
             result = jf!(result, xᵢ)
             x_old = copy(xᵢ)
             fᵢ[:], kᵢ[:,:] = result.value, result.derivs[1]
         catch error
             #println(error.msg)
             println(" jacobian and forward model calculation has failed")
             return failed_inversion(x₀, measurement)
         end
         

        # Gauss-Newton Algorithm
         xᵢ[:] = xᵢ + inv(kᵢ'* Sₑ⁻¹ *kᵢ)*kᵢ'* Sₑ⁻¹ *(y - fᵢ);

        #evaluate relative difference between this and previous iteration 
         δᵢ = abs((norm( fᵢ .- y) .- norm(f_old .- y)) ./ norm(f_old .- y));
         #δᵢ = abs(norm( x_old .- xᵢ) ./ norm(x_old));
         if inversion_setup["verbose_mode"]
            println("δᵢ for iteration ",i," is ",δᵢ)
        end
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
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

    verbose = inversion_setup["verbose_mode"]
    # define the observational prior error covariance
    if haskey(inversion_setup, "obs_covariance")
        if verbose; println("Using user-defined covariance"); end
        Sₑ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_windows")
        if verbose; println("masking out selected wave-numbers"); end
        Sₑ⁻¹ = make_obs_error(measurement, masked_windows=inversion_setup["masked_windows"])
    else
        if verbose; println("default covariance"); end
        Sₑ⁻¹ = make_obs_error(measurement)
    end
    

    # define the a priori covariance 
        if haskey(inversion_setup, "Sₐ⁻¹")
            if verbose; println("custom apriori matrix"); end
        Sₐ⁻¹ = inversion_setup["Sₐ⁻¹"]
        else
            if verbose; println("using default a priori covarience matrix"); end
        Sₐ⁻¹ = make_prior_error(inversion_setup["σ"]); # a priori covarience  matrix 
        end
    
    
    # state vectors 
    xₐ = assemble_state_vector!(x₀); # apriori
    
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
        if verbose
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
    f = generate_forward_model(xₐ, measurement, spectra, inversion_setup)
    results = try
        nonlinear_inversion(f, xₐ, measurement, spectra, inversion_setup)
    catch e
        println("Inversion for measurement ", measurement_num, " has failed.")
        rethrow(e)
        failed_inversion(xₐ, measurement)
    end    
    return results
end




function run_inversion(xₐ::AbstractDict, dataset::AbstractDataset, molecules::Array{MolecularMetaData,1}, inversion_setup::Dict, spectral_windows::Vector)
    
    num_measurements = length(dataset.pressure) # number of total measurements
    modelled = Array{InversionResults}(undef, num_measurements)
    num_windows = length(spectral_windows);
    results = Array{AbstractResults}(undef, (num_measurements, num_windows));
    println("Beginning inversion")
    
    for (j, spectral_window) in enumerate(spectral_windows)

        spectra = setup_molecules(molecules)
        out = pmap(i -> fit_spectra(i, xₐ, dataset, spectra, spectral_window, inversion_setup), 1:num_measurements)
        results[:,j] = out;
    end
    return results
end


function process_all_files(xₐ::AbstractDict,
                           molecules::Array{MolecularMetaData,1},
                           inversion_setup::Dict,
                           spectral_windows::AbstractArray;
                           data_path::String=pwd(),
                           out_path::String=pwd())
    
    files = readdir(data_path);
    num_files = length(files)
    
    for i=1:num_files
        
        file = files[i]
        full_file = joinpath(data_path, file)
        if endswith(file, ".h5") == false; continue; end;
        println(i,"/",num_files);
        println(file)
        
        data = read_DCS_data(full_file)
        data = take_time_average!(data, δt=inversion_setup["averaging_window"])
        results = run_inversion(xₐ, data, molecules, inversion_setup, spectral_windows)

        outfile = out_path*"/"*file[1:end-3]*"_results.jld2";
        #save_results(outfile, results, experiment_labels)
        @save outfile results
    end
    println("done with all files")
    return true
end


function process_all_files(xₐ::AbstractDict,
                           molecules::Array{MolecularMetaData,1},
                           inversion_setup::Dict,
                           spectral_windows::AbstractArray,
                           datafiles::Array{String,1};
                           data_path::String=pwd(),
                           out_path::String=pwd())

    num_files = length(datafiles)

    for i=1:num_files
        file = joinpath(data_path, datafiles[i])
        if endswith(file, ".h5") == false; continue; end;
        println(i,"/",num_files);
        println(file)
        
        data = read_DCS_data(file)
        data = take_time_average!(data, δt=inversion_setup["averaging_window"])
        results = run_inversion(xₐ, data, molecules, inversion_setup, spectral_windows)

        outfile = out_path*"/"*datafiles[i][1:end-3]*"_results.jld2";
        @save outfile results
    end
    println("done with all files")
    return true
end
