
using LinearAlgebra, SpectralFits

function construct_spectra(molecules::Array{MolecularMetaData,1}; p::Array{<:Real,1}=collect(range(400, 800, length=20)), T::Array{<:Real,1}=collect(range(240, 290, length=20)), ν_min=6000, ν_max=6400, δν=0.01)
    n_levels = length(p) 
    spectra = Array{OrderedDict}(undef, n_levels)
    for i=1:length(p)
        spectra[i] = SpectralFits.construct_spectra(molecules, p=p[i], T=T[i], ν_min=ν_min, δν=δν, ν_max=ν_max)
    end
    return spectra
end


function construct_spectra!(spectra::Array{OrderedDict,1}; p::Array{<:Real,1}=collect(range(450,800, length=20)), T::Array{<:Real,1}=collect(range(240, 290, length=20)))
    num_levels = length(p)
    for i=1:num_levels
        spectra[i] = SpectralFits.construct_spectra!(spectra[i], p=p[i], T=T[i])
    end
    return spectra
end

function half_pressure_levels(p::Array{<:Real,1})
    half_levels::Array{<:Real,1} = zeros(length(p)+1)
    p₀ = p[1]
    for i=2:length(p)
        half_levels[i] = (p[i] + p[i-1])/2
    end
    half_levels[1] = p₀ - (half_levels[2]-p₀)
    half_levels[end] = p[end] + (p[end]-half_levels[end-1])
    return half_levels
end


function assemble_state_vector!(x::Array{<:Real,1}, fields::Array{Any,1}, num_levels::Integer, inversion_setup::AbstractDict)
    out::OrderedDict{Any, Array{<:Real,1}} = OrderedDict([fields[i] => x[1+(i-1)*num_levels : i*num_levels] for i=1:length(fields)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end



function make_prior_error(σ::Union{Array{<:Real,1}, OrderedDict})
    if typeof(σ) <: AbstractDict
        σ = assemble_state_vector!(σ)
    end
    
    n = length(σ)
    Sₐ⁻¹ = zeros(n,n)
    x = 1 ./ σ.^2
   # bad_idx = findall(x.==Inf)
     #[x[i] = 1/1.0e10 for i in bad_idx]

    for i=1:n
        Sₐ⁻¹[i,i] = x[i]
    end
    return Sₐ⁻¹
end


               

    
function make_vcd_profile(p::Array{<:Real,1}, T::Array{<:Real,1}, vmr_H₂O::Array{<:Real,1})
    
    vcd = zeros(size(p))
    half_levels = half_pressure_levels(p)
    δp = half_levels[2:end] - half_levels[1:end-1]
    input_variables = zip(δp,T,vmr_H₂O)
    vcd = map(x -> SpectralFits.vcd_pressure(x[1], x[2], x[3]), input_variables)
    return vcd
end

    

function calculate_transmission(xₐ::AbstractDict, spectra::Array{<:AbstractDict,1})

    vcd = 2*make_vcd_profile(p, T, xₐ[H₂O])

    n_levels = length(p)
    τ = zeros(size(spectra[1][CO₂].grid))
    for i = 1:n_levels
        for species in keys(spectra[1])
            τ += vcd[i]*xₐ[species][i]*spectra[i][species].cross_sections
        end
    end
    return exp.(-τ)
end




function generate_profile_model(xₐ::AbstractDict, measurement::Measurement, spectra::Array{OrderedDict,1}, inversion_setup::AbstractDict)
    x_fields = collect(keys(xₐ))
    num_levels = length(xₐ[x_fields[1]])
    
    
    function f(x)

        if typeof(x) <: Array
            x = assemble_state_vector!(x, x_fields, num_levels, inversion_setup)
        end
        if inversion_setup["fit_pressure"]
            println("fitting p and T")
            p, T = x["pressure"], x["temperature"]
                    spectra = construct_spectra!(spectra, p=p, T=T)
        end
        

        transmission = calculate_transmission(x, spectra)
        transmission = apply_instrument(collect(spectra[1][H₂O].grid), transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline 
        shape_parameters = x["shape_parameters"]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity = transmission .* polynomial_term
        return intensity
    end
    return f
end


function nonlinear_inversion(f::Function, x₀::AbstractDict, measurement::Measurement, spectra::Array{<:AbstractDict,1}, inversion_setup::AbstractDict)
    

    Sₒ⁻¹ = SpectralFits.make_prior_error(measurement, a=0.0019656973992654737);
    
    #Sₑ = diagm(ones(length(measurement.intensity)));
    y = measurement.intensity;
    Kᵢ = zeros(length(y), length(x₀));
    xₐ = assemble_state_vector!(x₀);
    xᵢ = xₐ
    Sₐ⁻¹ = make_prior_error(inversion_setup["σ"])
    tolerence = 1.0e-4;
    γ = 1;
    δᵢ = 15;
    i = 1
    fᵢ = f(xᵢ)
    

    # begin the non-linear fit
    while i<15 && δᵢ>tolerence

        # evaluate the model and jacobian 
        result = DiffResults.JacobianResult(zeros(length(collect(measurement.grid))), xᵢ);
        ForwardDiff.jacobian!(result, f, xᵢ);
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
        println("δᵢ for iteration ",i," is ",δᵢ)        
        i = i+1
    end #while loop

    # Calculate χ²
    χ² = (y-fᵢ)'*Sₒ⁻¹*(y-fᵢ)/(length(fᵢ)-length(xᵢ))
    S = inv(Kᵢ'*Sₒ⁻¹*Kᵢ); # posterior error covarience

    # Gain matrix
    G = inv(Kᵢ'*Sₒ⁻¹*Kᵢ + Sₐ⁻¹)*Kᵢ'*Sₒ⁻¹
    return InversionResults(measurement.time, xᵢ, y, fᵢ, χ², S, measurement.grid, G, Kᵢ, Sₒ⁻¹, Sₐ⁻¹)
end#function
