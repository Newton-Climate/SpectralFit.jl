
using Interpolations, Statistics



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
    k = collect(keys(spectra))
    τ = zeros(size(spectra[k[1]].cross_sections));
    for key in keys(spectra)
        τ += vcd * x[key] * spectra[key].cross_sections
        end
    return exp.(-τ)
end


"""calculate transmission in a profile with multiple layers"""
function calculate_transmission(xₐ::AbstractDict, spectra::Array{<:AbstractDict,1}, vcd::Array{<:Real,1})
   
    n_levels = length(vcd)
    k = collect(keys(spectra[1]))
    τ = zeros(size(spectra[1][k[1]].grid))
    for i = 1:n_levels
        for species in keys(spectra[1])
            τ += vcd[i]*xₐ[species][i]*spectra[i][species].cross_sections
        end
    end
    return exp.(-τ)
end




"""
down-scales the co-domain of spectral cross-sections grid to the instrument grid
"""
function apply_instrument( input_spectral_grid::Array{<:Real,1},
                               input_spectra::Array{<:Real,1},
                           output_spectral_grid::Array{<:Real,1})
    
    δν = mean(diff(input_spectral_grid));
    ν_min, ν_max = input_spectral_grid[1], input_spectral_grid[end];
    ν = ν_min:δν:ν_max;
    
    itp = interpolate(input_spectra, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, ν)
    return sitp(output_spectral_grid)
end # end of function apply_instrument



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
#        if inversion_setup["use_OCO"] && spectra[CO₂].grid[1] >= 6140 && spectra[CO₂].grid[end] <= 6300
#        println("fitting temperature with OCO database")
#            spectra[CO₂].cross_sections = OCO_interp(spectra[CO₂].grid, x["temperature"], x["pressure"])
#        end
        
        # apply Beer's Law
        transmission = calculate_transmission(x, measurement, spectra)

        # down-sample to instrument grid 
        transmission = apply_instrument(collect(spectra[x₀_fields[1]].grid), transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline 
        shape_parameters = x["shape_parameters"]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity = transmission .* polynomial_term
        return intensity
    end
    return f
end


"""Generate a foreward model that calculates the transmission through multiple layers of the atmosphere"""
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
                    spectra = construct_spectra_by_layer!(spectra, p=p, T=T)
        end
        
        transmission = calculate_transmission(x, spectra, measurement.vcd)
        transmission = apply_instrument(collect(spectra[1][x_fields[1]].grid), transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline 
        shape_parameters = x["shape_parameters"]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity = transmission .* polynomial_term
        return intensity
    end
    return f
end


