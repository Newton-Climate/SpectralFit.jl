using LinearAlgebra, Plots, Distributed, DistributedArrays, DiffResults
using Debugger
include("read_data.jl")
include("forward_model.jl")
using Plots



function make_prior_error(measurement::Measurement; a::Float64=0.3*0.01611750368314077)
    n = length(measurement.intensity)
    Sₑ = zeros((n,n))
    base = mean(measurement.intensity)
    for i = 1:n
        Sₑ[i,i] = 1/(a*sqrt(base))^2
    end
    return Sₑ
end

function make_prior_error(dataset::Dataset; a::Float64=0.3*0.01611750368314077)
    n = length(measurement.intensity)
    Sₑ = zeros((n,n))
    base = mean(dataset.intensity)
    for i = 1:n
        Sₑ[i,i] = 1/(a*sqrt(base))^2
    end
    return Sₑ
end


function nonlinear_inversion(x₀::Array{<:Real,1}, measurement::Measurement, spectra::Spectra, inversion_setup::Dict{String,Real})
    f = generate_forward_model(measurement, spectra, inversion_setup);
    Sₑ = make_prior_error(measurement);
    #Sₑ = diagm(ones(length(measurement.intensity)));
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
#        @show x[1:4]

        xᵢ = x;
        δᵢ = abs((norm( fᵢ - y) - norm(f_old - y)) / norm(f_old - y));
        
        if i==1 #prevent premature ending of while loop
            δᵢ = 1
        end
        
        println("δᵢ for iteration ",i," is ",δᵢ)        
        i = i+1
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₑ*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(kᵢ'*Sₑ*kᵢ)
    return InversionResults(measurement.time, xᵢ, y, fᵢ, χ², S)
end#function

    

function fit_spectra(measurement_num::Integer, xₐ::Array{<:Real,1}, dataset::Dataset, ν_range::Tuple)
    measurement = get_measurement(measurement_num, dataset, ν_range[1], ν_range[2])
    spectra = construct_spectra("../H2O_S.data", "../CH4_S.data", "../CO2_S.data", "../HDO_S.data", ν_min=ν_range[1]-3, ν_max=ν_range[2]+3, p=measurement.pressure, T=measurement.temperature)
    results = try
        nonlinear_inversion(xₐ, measurement, spectra, inversion_setup)
    catch
        InversionResults(measurement.time, NaN, NaN, NaN, NaN, NaN)
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
    
    #    Threads.@threads for i=1:num_measurements
    for i = 1:num_measurements
        
        if i%100 == 0
            println(i)
        end
        println(i)
        results[1,i] = fit_spectra(i, xₐ, dataset, ν_CO₂);        
        results[2,i] = fit_spectra(i, xₐ, dataset, ν_CH₄);
        results[3,i] = fit_spectra(i, xₐ, dataset, ν_HDO);
    end
    return results
end

        
function invert_all_files(xₐ::Array{<:Real}, inversion_setup::Dict; path=pwd())
    files = readdir(path);
    num_files = length(files)
    
    Threads.@threads for i=1:num_files

        file = files[i];
        if endswith(file, ".h5") == false; continue; end;
        println(i,"/",num_files);
        data = read_new_NIST_data(file);

        if inversion_setup["take_time_average"]
            data = take_time_average(data)
        end

        results = run_inversion(xₐ, data, inversion_setup)

        outfile = file[1:end-3]*"_results.h5";
        save_inversion_results(outfile, results, data, ["CO2_band", "CH4_band", "H2O_band"]);
    end
    println("done with all files")
    return true
end
