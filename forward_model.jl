include("read_data.jl")
using Interpolations, Statistics
using ForwardDiff

global const H₂O_ind, CH₄_ind, CO₂_ind, HDO_ind = 1, 2, 3, 4;
global const temperature_ind, pressure_ind = 5, 6
global const windspeed_ind = 7
global OCO_interp = OCO_spectra("../../co2_v5.1_wco2scale=nist_sco2scale=unity.hdf")

function calculate_transmission(x::Array{<:Real,1}, measurement::Measurement, spectra::Spectra)
    """
Calculates the transmission given Beer's Law

- Arguements
    1. xₐ::Vector: The state vector containing the parametres to fit
2. measurement::Measurement: a Measurement type subsetted from the Dataset type
3. spectra::Spectra: A Spectra type containing the Molecules type and thus cross-sections
4. inversion_setup::AbstractDict: A dictionary contaiing the flags for the inversion

- returns:
transmission::Vector: the calculated tranmission
"""

    vcd = measurement.vcd
    H₂O_vmr, H₂O_cs = x[H₂O_ind], spectra.H₂O.cross_sections
    CH₄_vmr, CH₄_cs = x[CH₄_ind], spectra.CH₄.cross_sections
    CO₂_vmr, CO₂_cs = x[CO₂_ind], spectra.CO₂.cross_sections
    HDO_vmr, HDO_cs = x[HDO_ind], spectra.HDO.cross_sections
    τ = vcd*(H₂O_vmr.*H₂O_cs .+ CH₄_vmr.*CH₄_cs .+ CO₂_vmr.*CO₂_cs .+ HDO_vmr.*HDO_cs)
    return exp.(-τ)
end

function calculate_transmission(x::AbstractDict, measurement::Measurement, spectra::AbstractDict)
    """
    Calculates the transmission given Beer's Law

- Arguements
    1. xₐ::AbstractDict: The state vector containing the parametres to fit
2. measurement::Measurement: a Measurement type subsetted from the Dataset type
3. spectra::AbstractDict: A dictionary containing the Molecules type and thus cross-sections
4. inversion_setup::AbstractDict: A dictionary contaiing the flags for the inversion

- returns:
transmission::Vector: the calculated tranmission
"""
    
    vcd = measurement.vcd
    τ = zeros(size(spectra[CO₂].cross_sections));
    for key in keys(spectra)
        τ += vcd * x[key] * spectra[key].cross_sections
        end
    return exp.(-τ)
end


function apply_instrument( input_spectral_grid::Array{<:Real,1},
                               input_spectra::Array{<:Real,1},
                           output_spectral_grid::Array{<:Real,1})
    """
down-scales the co-domain of spectral cross-sections grid to the instrument grid
"""
    

    δν = mean(diff(input_spectral_grid));
    ν_min, ν_max = input_spectral_grid[1], input_spectral_grid[end];
    ν = ν_min:δν:ν_max;
    
    itp = interpolate(input_spectra, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, ν)
    output_spectra = sitp(output_spectral_grid)
       return output_spectra
#    return itp(output_spectral_grid)
end # end of function apply_instrument

function compute_legendre_poly(x::Array{<:Real,1}, nmax::Integer)
    """
calculates the legendre polynomial over domain x::Vector of degree max::Integer
"""
    
    FT = eltype(x)
    @assert nmax > 1
    #@assert size(P) == (nmax,length(x))
    P⁰ = zeros(nmax,length(x));
   
    # 0th Legendre polynomial, a constant
    P⁰[1,:] .= 1;

    # 1st Legendre polynomial, x
    P⁰[2,:] = x;

    for n=2:nmax-1
        for i in eachindex(x)
            l = n-1
            P⁰[n+1,i] = ((2l + 1) * x[i] * P⁰[n,i] - l * P⁰[n-1,i])/(l+1)
        end
    end
    return P⁰
end  

function calc_polynomial_term( legendre_polynomial_degree::Integer,
                             shape_parameters::Array{<:Real,1},
                               wavenumber_grid_length::Integer)
    """
calculates the Legendre Polynomial Coefficients and weights by the shape parameters
"""
    
    
    x = collect(range(-1, stop=1, length=wavenumber_grid_length));
    polynomial_term = Array{Float64}(undef, length(x))
    polynomial_term = shape_parameters' * compute_legendre_poly(x, legendre_polynomial_degree)
    return polynomial_term
end

function DopplerShift( relative_speed::Real, spectral_grid::Array{<:Real,1})
    # calculate shift
    shift1 = (relative_speed/c) * spectral_grid
    shift2 = -(relative_speed/c)*spectral_grid
    
    coef1 = ones( size(spectral_grid)) + shift1
    coef2 = ones(size(spectral_grid)) + shift2
    spectral_grid_out = coef1 * coef2 * spectral_grid
    return spectral_grid_out
end

function assemble_state_vector!(x::AbstractDict)
    out::Array{Real,1} = []
    for key in keys(x)
        out = append!(out, x[key])
        end
    return out
end

function assemble_state_vector!(x::Vector{<:Real}, key_vector::Array{Any,1}, inversion_setup::AbstractDict)
    out::OrderedDict{Any,Any} = OrderedDict([key_vector[i] => x[i] for i=1:length(key_vector)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end

function fit_pressure!(x::AbstractDict, measurement::Measurement, spectra::AbstractDict, inversion_setup::Dict)
    if inversion_setup["use_OCO"] == false && inversion_setup["use_TCCON"] == false
        println("inside pressure function")
        println(typeof(x))
        println(typeof(x["pressure"]))
        spectra = construct_spectra!(spectra, p=x["pressure"], T=measurement.temperature)
    elseif inversion_setup["use_OCO"] && spectra[CO₂].grid[1] >= 6140 && spectra[CO₂].grid[end] <= 6300
        println("fitting pressure with OCO database")
        spectra[CO₂].cross_sections = OCO_interp(spectra[CO₂].grid, measurement.temperature, x["pressure"])
    else
        println("This wavenubmer range is out of the OCO or TCCON database range. Using HiTran")
        spectra = construct_spectra!(spectra, p=x["pressure"], T=measurement.temperature)
    end
    return spectra
end


function fit_temperature!(x::AbstractDict, measurement::Measurement, spectra::AbstractDict, inversion_setup::Dict)
    if inversion_setup["use_OCO"] == false && inversion_setup["use_TCCON"] == false
        spectra = construct_spectra!(spectra, p=measurement.pressure, T=x["temperature"])
    elseif inversion_setup["use_OCO"] && spectra[CO₂].grid[1] >= 6140 && spectra[CO₂].grid[end] <= 6300
        println("fitting temperature with OCO database")
        spectra[CO₂].cross_sections = OCO_interp(spectra[CO₂].grid, x["temperature"], measurement.pressure)
    else
        println("This wavenubmer range is out of the OCO or TCCON database range. Using HiTran")
        spectra = construct_spectra!(spectra, p=measurement.pressure, T=x["temperature"])
    end
    return spectra
end

function fit_pT!(x::AbstractDict, measurement::Measurement, spectra::AbstractDict, inversion_setup::Dict)
    if inversion_setup["use_OCO"] == false && inversion_setup["use_TCCON"] == false
        spectra = construct_spectra!(spectra, p=x["pressure"], T=x["temperature"])
    elseif inversion_setup["use_OCO"] && spectra[CO₂].grid[1] >= 6140 && spectra[CO₂].grid[end] <= 6300
        println("fitting temperature with OCO database")
        spectra[CO₂].cross_sections = OCO_interp(spectra[CO₂].grid, x["temperature"], x["pressure"])
    else
        println("This wavenubmer range is out of the OCO or TCCON database range. Using HiTran")
        spectra = construct_spectra!(spectra, p=x["pressure"], T=x["temperature"])
    end
    return spectra
end

        

function generate_forward_model(measurement::Measurement, spectra::Spectra, inversion_setup::Dict{String,Real})
    #OCO_interp = OCO_spectra("../../co2_v5.1_wco2scale=nist_sco2scale=unity.hdf")
    
    function f(x)
#        OCO_interp = OCO_spectra("../../co2_v5.1_wco2scale=nist_sco2scale=unity.hdf")
#        println(x[temperature_ind])
#                spectra.CO₂.cross_sections = OCO_interp(spectra.CO₂.grid, x[temperature_ind], x[pressure_ind])
        # println(typeof(x))

        if inversion_setup["fit_pT"] && inversion_setup["use_OCO"] == false
            spectra = construct_spectra!(spectra, p=x[pressure_ind], T=x[temperature_ind])
        elseif inversion_setup["fit_pT"] && inversion_setup["use_OCO"] && spectra.CO₂.grid[1] >= 6140 && spectra.CO₂.grid[end] <= 6300
            spectra.CO₂.cross_sections = OCO_interp(spectra.CO₂.grid, x[temperature_ind], x[pressure_ind])
        end
        
        transmission = calculate_transmission(x, measurement, spectra)
        transmission = apply_instrument(collect(spectra.HDO.grid), transmission, measurement.grid)
        degree = inversion_setup["poly_degree"];
        shape_parameters = x[end - degree+1 : end]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity = transmission .* polynomial_term
        return intensity
    end
    return f
end


function generate_forward_model(x₀::AbstractDict, measurement::Measurement, spectra::AbstractDict, inversion_setup::AbstractDict)
    """
- generates a forward model given the inversion parameters
- Called as follows:
f = generate_forward_model(xₐ, measurement, spectra, inversion_setup)
intensity = f(x) where x<:Array

- Arguements
    1. xₐ::AbstractDict: The state vector containing the parametres to fit
2. measurement::Measurement: a Measurement type subsetted from the Dataset type
3. spectra::AbstractDict: A dictionary containing the Molecules type and thus cross-sections
4. inversion_setup::AbstractDict: A dictionary contaiing the flags for the inversion

Returns:
f::Function: the forward model called as f(x::Vector)
"""
    

    # Save the labelled fields in the state vector
    x₀_fields = collect(keys(x₀))
    
    function f(x)
        # convert the state vector to a dict with labelled fields
        x = assemble_state_vector!(x, x₀_fields, inversion_setup)

        # update the cross-sections given pressure and temperature 
        spectra = construct_spectra!(spectra, p=x["pressure"], T=x["temperature"])

        #for the OCO line-list for CO₂
        if inversion_setup["use_OCO"] && spectra[CO₂].grid[1] >= 6140 && spectra[CO₂].grid[end] <= 6300
        println("fitting temperature with OCO database")
            spectra[CO₂].cross_sections = OCO_interp(spectra[CO₂].grid, x["temperature"], x["pressure"])
        end
        
        # apply Beer's Law
        transmission = calculate_transmission(x, measurement, spectra)

        # down-sample to instrument grid 
        transmission = apply_instrument(collect(spectra[CO₂].grid), transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline 
        shape_parameters = x["shape_parameters"]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity = transmission .* polynomial_term
        return intensity
    end
    return f
end
