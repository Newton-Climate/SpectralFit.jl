
#OCO_path = joinpath(dirname(pathof(SpectralFits)), "..", "CrossSections_data", "OCO_spectra.hdf")
#global OCO_interp = OCO_spectra(OCO_path)


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
function calculate_transmission(x::AbstractDict, pathlength::Real, spectra::AbstractDict, grid_length::Int; p::Real=1000, T::Real=290)

    FT = dicttype(x)
    τ = zeros(FT, grid_length)
    vcd = calc_vcd(p, T, pathlength)
    
    for molecule in keys(spectra)
        σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
    τ += vcd * x[molecule] * σ
    end
    return exp.(-τ)
end

function calculate_transmission(x::AbstractDict, spectra::AbstractDict, grid_length::Int; p::Real=1000, T::Real=300)

    FT = dicttype(x)
    τ = zeros(FT, grid_length)
    
    for molecule in keys(spectra)
        σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
    τ += x[molecule] * σ
    end
    return exp.(-τ)
end


"""calculate transmission in a profile with multiple layers"""
function calculate_transmission(xₐ::AbstractDict, spectra::AbstractDict, p::Vector{<:Real}, T::Vector{<:Real}; input_is_column=false)                                
    


    n_levels = length(p)
    vcd = input_is_column ? ones(n_levels) : make_vcd_profile(p, T)
    molecules = collect(keys(spectra))
    FT = dicttype(xₐ)
    τ = zeros(FT, length(spectra[molecules[1]].grid))
    
    for i = 1:n_levels
        for molecule in molecules
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p[i], T[i])
            τ += vcd[i]*xₐ[molecule][i]*σ
        end
    end
    return exp.(-τ)
end




"""
down-scales the co-domain of spectral cross-sections grid to the instrument grid
"""
function apply_instrument( input_spectral_grid::AbstractArray{<:Real},
                               input_spectra::AbstractArray{<:Real},
                           output_spectral_grid::AbstractArray{<:Real})
    
    itp = interpolate(input_spectra, BSpline(Quadratic(Line(OnGrid()))))
    #itp = interpolate(input_spectra, BSpline(Cubic(Line(OnGrid()))))
    sitp = scale(itp, input_spectral_grid)
    return sitp(output_spectral_grid)
end # end of function apply_instrument




function calc_polynomial_term( legendre_polynomial_degree::Integer,
                                shape_parameters::Array{<:Real,1},
                                wavenumber_grid_length::Integer)
    """
calculates the Legendre Polynomial Coefficients and weights by the shape parameters
"""
                
    x = collect(range(-1, stop=1, length=wavenumber_grid_length));
    poly_matrix = compute_legendre_poly(x, legendre_polynomial_degree)
   return  transpose(shape_parameters' * poly_matrix)
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



function fit_pressure!(x::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::Dict)
    if inversion_setup["use_OCO"] == false && inversion_setup["use_TCCON"] == false
        println("inside pressure function")
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


function fit_temperature!(x::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::Dict)
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

function fit_pT!(x::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::Dict)
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




function generate_forward_model(x₀::AbstractDict, measurement::FrequencyCombMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)
    """
- generates a forward model given the inversion parameters
- Called as follows:
f = generate_forward_model(xₐ, measurement, spectra, inversion_setup)
intensity = f(x) where x<:Array

- Arguements
    1. xₐ::AbstractDict: The state vector containing the parametres to fit
2. measurement::AbstractMeasurement: a Measurement type subsetted from the Dataset type
3. spectra::AbstractDict: A dictionary containing the Molecules type and thus cross-sections
4. inversion_setup::AbstractDict: A dictionary contaiing the flags for the inversion

Returns:
f::Function: the forward model called as f(x::Vector)
"""
    

    # Save the labelled fields in the state vector
    x_fields = collect(keys(x₀))
    
    function f(x)
        # convert the state vector to a dict with labelled fields
        if typeof(x) <: AbstractArray
            x = assemble_state_vector!(x, x_fields, inversion_setup)
        end
        
        spectra_grid = spectra[x_fields[1]].grid
        len_spectra = length(spectra_grid)
        len_measurement = length(measurement.grid)
                
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature

        # apply Beer's Law
        if haskey(inversion_setup, "fit_column") && inversion_setup["fit_column"] == true
            transmission = calculate_transmission(x, spectra, len_spectra, p=p, T=T)
        else
            transmission = calculate_transmission(x, measurement.pathlength, spectra, len_spectra, p=p, T=T)
        end

        # down-sample to instrument grid
        intensity = apply_instrument(spectra_grid, transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline
        if inversion_setup["linear"]
            intensity = log.(intensity)
            intensity .+= calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], len_measurement)
        else
            intensity .*= calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], len_measurement)
        end
        return intensity 
    end
    return f
end


"""Generate a foreward model that calculates the transmission through multiple layers of the atmosphere"""
function generate_profile_model(xₐ::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, inversion_setup::AbstractDict)
    x_fields = collect(keys(xₐ))
    num_levels = length(xₐ[x_fields[1]])
    
    
    function f(x)

        if typeof(x) <: Array
            x = assemble_state_vector!(x, x_fields, num_levels, inversion_setup)
        end

                spectra_grid = spectra[x_fields[1]].grid
        len_spectra = length(spectra_grid)
        len_measurement = length(measurement.grid)
                
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
        
        transmission = calculate_transmission(x, spectra, p, T, input_is_column=inversion_setup["fit_column"])
        intensity = apply_instrument(spectra_grid, transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline
        if inversion_setup["linear"]
            intensity = log.(intensity)
            intensity .+= calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], len_measurement)
        else
            intensity .*= calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], len_measurement)
        end
            
        return intensity 
    end
    return f
end

