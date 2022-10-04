
function make_obs_error(measurement::AbstractMeasurement;
                        σ²::Union{Nothing, Float64}=nothing,
                        masked_windows::Union{AbstractArray{<:Real}, Nothing}=nothing,
                        linear=false)
    
    n = length(measurement.intensity)

    if σ²==nothing # get noise from the instrument
        noise = measurement.σ²
    else #get noise from user 
        noise = σ²
    end

    value = ones(n)
    if linear
        println("linear obs error")
        value .= 1 ./ (noise ./ measurement.intensity .^2)
    else
        value .*= @. 1/noise
    end
    
    
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
    linear = inversion_setup["linear"]
    if haskey(inversion_setup, "obs_covariance")
        if verbose; println("Using user-defined covariance"); end
        Sₑ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_windows")
        if verbose; println("masking out selected wave-numbers"); end
        Sₑ⁻¹ = make_obs_error(measurement, masked_windows=inversion_setup["masked_windows"], linear=linear)
    else
        if verbose; println("default covariance"); end
        Sₑ⁻¹ = make_obs_error(measurement, linear=linear)
    end
    
    y = measurement.intensity;
    xᵢ = x₀;
    xᵢ = assemble_state_vector!(xᵢ)
    tolerence = 1.0e-2;
    δᵢ = 10.0;
    i = 1
    max_iter = linear ? 1 : 30
    state_length, grid_length = length(xᵢ), length(measurement.grid)
        kᵢ = zeros(grid_length, state_length)
        fᵢ = zeros(grid_length)
        f_old = similar(fᵢ)
        chunk_size = state_length < 30 ? state_length : 20
    #cfg = ForwardDiff.JacobianConfig(f,xᵢ, ForwardDiff.Chunk{chunk_size}())
    result = DiffResults.JacobianResult(measurement.grid, xᵢ);
    jf! = (out, _x) -> ForwardDiff.jacobian!(out, f, _x)
    
    # begin the non-linear fit
     while i <= max_iter && δᵢ>tolerence

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
         χ² = (y-fᵢ)'* Sₑ⁻¹ *(y-fᵢ)/(length(fᵢ)-length(xᵢ))
         #δᵢ = abs(norm( x_old .- xᵢ) ./ norm(x_old));
         if inversion_setup["verbose_mode"]
             println("δᵢ for iteration ",i," is ",δᵢ)
             @show χ²
        end
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
            end

         i += 1
         f_old, δ_old = fᵢ, δᵢ
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
    linear = inversion_setup["linear"]
    # define the observational prior error covariance
    if haskey(inversion_setup, "obs_covariance")
        if verbose; println("Using user-defined covariance"); end
        Sₒ⁻¹ = inversion_setup["obs_covarience"] 
    elseif haskey(inversion_setup, "masked_windows")
        if verbose; println("masking out selected wave-numbers"); end
        Sₒ⁻¹ = make_obs_error(measurement, masked_windows=inversion_setup["masked_windows"], linear=linear)
    else
        if verbose; println("default covariance"); end
        Sₒ⁻¹ = make_obs_error(measurement, linear=linear)
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
    
    tolerence = 1.0e-3; # relative error reached to stop loop

    #regularization parameter 
    if haskey(inversion_setup, "γ")
        γ = inversion_setup["γ"]
        else
        γ = 1.0; # regularization parameter
    end
    
    δᵢ, δ_old = 15.0, 20.0; # relative errror
    χ²_old = 0.001
    i = 1; # iteration count
    max_iter = linear ? 1 : 30

    # allocate memory for inversion matrixes
    y = measurement.intensity; # obserbations 
    Kᵢ = zeros((length(measurement.grid), length(xᵢ))) # jacobian
    f_old = similar(y) # previous model-run 
    fᵢ = similar(y) # current model-run
    m, n = length(y), length(xᵢ)
    degrees::Float64 = m - n

    # begin the non-linear fit
    while i <= max_iter && δᵢ>tolerence

        # evaluate the model and jacobian 
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), xᵢ);
        ForwardDiff.jacobian!(result, f, xᵢ);
        fᵢ, Kᵢ = result.value, result.derivs[1]
        x_old = xᵢ

        # Baysian Maximum Likelihood Estimation 
        # lhs = (Sₐ⁻¹ + Kᵢ'*Sₐ⁻¹*Kᵢ + γ*Sₒ⁻¹)
        lhs = (Kᵢ'*Sₒ⁻¹*Kᵢ + Sₐ⁻¹ + γ*Sₐ⁻¹)
        rhs = (Kᵢ'*Sₒ⁻¹ * (y - fᵢ) - Sₐ⁻¹*(xᵢ - xₐ))
        Δx = lhs\rhs
        xᵢ = xᵢ + Δx; # reassign state vector for next iteration

        #evaluate relative difference between this and previous iteration 
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));
         χ² = (y-fᵢ)'* Sₒ⁻¹ *(y-fᵢ)/degrees 
         #δᵢ = abs(norm( x_old .- xᵢ) ./ norm(x_old));
         if inversion_setup["verbose_mode"]
             println("δᵢ for iteration ",i," is ",δᵢ)
             @show χ²
        end
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
        end
        if χ²_old < 1.1 && δᵢ > 1.0; xᵢ = x_old; break; end


#        if 0.7 < χ² / χ²_old < 1.0
#            γ *= 0.5
#            @show γ
#        end
        

       
         i += 1
        f_old, δ_old = fᵢ, δᵢ
        χ²_old = χ²
#        δᵢ > 1.0 ? xᵢ = xₐ : continue
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ)/degrees
    S = inv(Kᵢ'*Sₒ⁻¹*Kᵢ); # posterior error covarience
    x=assemble_state_vector!(xᵢ, collect(keys(x₀)), num_levels, inversion_setup)

    # Gain matrix
    return InversionResults(timestamp=measurement.time, machine_time=measurement.machine_time,
                              x=x,
                              measurement=y, model=fᵢ, χ²=χ², S=S,
                              grid=measurement.grid, K=Kᵢ, Sₑ⁻¹=Sₒ⁻¹, Sₐ⁻¹=Sₐ⁻¹)
end#function


function adaptive_inversion(f::Function, x₀::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)

    verbose = inversion_setup["verbose_mode"]
    linear = inversion_setup["linear"]
    # define the observational prior error covariance
    if haskey(inversion_setup, "obs_covariance")
        if verbose; println("Using user-defined covariance"); end
        Sₒ⁻¹ = inversion_setup["obs_covarience"]
    elseif haskey(inversion_setup, "masked_windows")
        if verbose; println("masking out selected wave-numbers"); end
        Sₒ⁻¹ = make_obs_error(measurement, masked_windows=inversion_setup["masked_windows"], linear=linear)
    else
        if verbose; println("default covariance"); end
        Sₒ⁻¹ = make_obs_error(measurement, linear=linear)
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
    max_iter = linear ? 10 : 30
    
    #regularization parameter 
    if haskey(inversion_setup, "γ")
        γ = inversion_setup["γ"]
        else
        γ = 1.0; # regularization parameter
    end
    
    δᵢ, δ_old = 15.0, 20.0; # relative errror
    χ²_old = 0.001
    i = 1; # iteration count 

    # allocate memory for inversion matrixes
    y = measurement.intensity; # obserbations 
    Kᵢ = zeros((length(measurement.grid), length(xᵢ))) # jacobian
    f_old = similar(y) # previous model-run 
    fᵢ = similar(y) # current model-run
    m, n = length(y), length(xᵢ)
    degrees::Float64 = m - n

    # begin the non-linear fit
    while i <= max_iter && δᵢ>tolerence

        # evaluate the model and jacobian 
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), xᵢ);
        ForwardDiff.jacobian!(result, f, xᵢ);
        fᵢ, Kᵢ = result.value, result.derivs[1]
        j_old = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ) + (xᵢ - xₐ)'*Sₐ⁻¹*(xᵢ - xₐ)
        x_old, f_old = xᵢ, fᵢ
        if linear
            measurement.intensity = fᵢ;
            Sₒ⁻¹ = make_obs_error(measurement, linear=linear)
        end
        # Baysian Maximum Likelihood Estimation 
        # lhs = (Sₐ⁻¹ + Kᵢ'*Sₐ⁻¹*Kᵢ + γ*Sₒ⁻¹)
        lhs = (Kᵢ'*Sₒ⁻¹*Kᵢ + Sₐ⁻¹ + γ*Sₐ⁻¹)
        rhs = (Kᵢ'*Sₒ⁻¹ * (y - fᵢ) - Sₐ⁻¹*(xᵢ - xₐ))
        δx = lhs\rhs
        xᵢ = xᵢ + δx; # reassign state vector for next iteration

        ## evaluate linearity of this step
        fᵢ = f(xᵢ)
        j_new = (y-fᵢ)'* Sₒ⁻¹ *(y-fᵢ) + (xₐ - xᵢ)'*Sₐ⁻¹*(xₐ - xᵢ)
        j_pred = (y - f_old - Kᵢ*δx)' *Sₒ⁻¹ *(y-f_old - Kᵢ*δx) + (xₐ - xᵢ - δx)'*Sₐ⁻¹*(xₐ - xᵢ - δx)
        r = (j_new - j_old) / (j_pred - j_old)
        
        # If not close to linear, then increase γ and step-size
        if r > 0.75
            γ = γ / 2.0
        elseif r < 0.25
            γ = γ <= 0 ? 1.0 : γ*10.0
        end
        @show γ
        @show r
        
        #evaluate relative difference between this and previous iteration 
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));
         χ² = (y-fᵢ)'* Sₒ⁻¹ *(y-fᵢ)/degrees 
         #δᵢ = abs(norm( x_old .- xᵢ) ./ norm(x_old));
         if inversion_setup["verbose_mode"]
             println("δᵢ for iteration ",i," is ",δᵢ)
             @show χ²
        end
        if i==1 #prevent premature ending of while loop
            δᵢ = 1.0
        end

        if 0.95 < χ² < 1.05; break; end
       
         i += 1
        δ_old = δᵢ
        χ²_old = χ²
    end #while loop

    # Calculate χ²      
    χ² = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ)/degrees
    S = inv(Kᵢ'*Sₒ⁻¹*Kᵢ  + Sₐ⁻¹); # posterior error covarience
    x=assemble_state_vector!(xᵢ, collect(keys(x₀)), num_levels, inversion_setup)

    # Gain matrix
    return InversionResults(timestamp=measurement.time, machine_time=measurement.machine_time,
                              x=x,
                              measurement=y, model=fᵢ, χ²=χ², S=S,
                              grid=measurement.grid, K=Kᵢ, Sₑ⁻¹=Sₒ⁻¹, Sₐ⁻¹=Sₐ⁻¹)
end#function
 
function fit_spectra(measurement_num::Integer, f::Function, xₐ::AbstractDict, dataset::AbstractDataset, spectra::AbstractDict, ν_range::Tuple, inversion_setup::Dict{String,Any})
    
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
        measurement = get_measurement(1, dataset, spectral_window[1], spectral_window[end])
    f = generate_forward_model(xₐ, measurement, spectra, inversion_setup)
        out = pmap(i -> fit_spectra(i, f, xₐ, dataset, spectra, spectral_window, inversion_setup), 1:num_measurements)
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
