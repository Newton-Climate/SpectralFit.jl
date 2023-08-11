
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
function calculate_transmission(x::AbstractDict, spectra::AbstractDict, pathlength::Real, grid_length::Int; 
                                p::Real=1000, T::Real=290, use_OCO_table=false, adjust_ch4_broadening=false, fit_column=false)

    FT = dicttype(x)
    τ = zeros(FT, grid_length)
    H2O = haskey(x, "H2O") ? x["H2O"] : 0.0
    vcd_wet = calc_vcd(p, T, pathlength)

    if fit_column
        vmr_H2O = H2O / (vcd_wet - H2O)
        vcd_dry = 1.0
    else
        vmr_H2O = x["H2O"]
        vcd_dry = calc_vcd(p, T, pathlength, vmr_H2O)
    end
    
   
    for molecule in keys(spectra)
 
        if spectra[molecule].molecule_num == 6 && adjust_ch4_broadening
        println("adjusting CH4 pressure")
            p_adjusted = p*(1+0.34*vmr_H2O)
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p_adjusted, T)

        elseif spectra[molecule].molecule_num == 2 && use_OCO_table
            println("using CO2 tables")
            σ = spectra[molecule].model.itp(spectra[molecule].grid, p, T, vmr_H2O)

        else
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
        end

    τ += x[molecule] * σ
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
function calculate_transmission(xₐ::AbstractDict, spectra::AbstractDict, p::Vector{<:Real}, T::Vector{<:Real}; input_is_column=false, sza=0.0)                                
    


    n_levels = length(p)
    vcd = input_is_column ? ones(n_levels) : make_vcd_profile(p, T)
    molecules = collect(keys(spectra))
    FT = dicttype(xₐ)
    τ = zeros(FT, length(spectra[molecules[1]].grid))
    amf = 1.0/cosd(sza)
    
    for i = 1:n_levels
        for molecule in molecules
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p[i], T[i])
            τ += vcd[i]*xₐ[molecule][i]*σ*amf
        end
    end
    
    return exp.(-τ)
end


"""calculate transmission in a profile with multiple layers"""
function calculate_transmission(xₐ::AbstractDict, spectra::AbstractDict, p::Vector{<:Real}, T::Vector{<:Real}, spectral_grid::AbstractArray; input_is_column=false)                                

    n_levels = length(p)
    vcd = input_is_column ? ones(n_levels) : make_vcd_profile(p, T)
    molecules = collect(keys(spectra))
    FT = dicttype(xₐ)
    τ = zeros(FT, length(spectral_grid))
    
    for i = 1:n_levels
        for molecule in molecules
            σ = absorption_cross_section(spectra[molecule].model, spectral_grid, p[i], T[i])
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
    
    #itp = interpolate(input_spectra, BSpline(Quadratic(Line(OnGrid()))))
    sitp = interpolate(input_spectra, BSpline(Cubic(Line(OnGrid()))))
    #sitp = scale(itp, input_spectral_grid)
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
        use_OCO_table = check_param(inversion_setup, "use_OCO_table")
        adjust_ch4_broadening = check_param(inversion_setup, "adjust_ch4_broadening")
        fit_column = check_param(inversion_setup, "fit_column")



                
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature

        # apply Beer's Law
        transmission = calculate_transmission(x, spectra, measurement.pathlength, len_spectra,
                                              p=p, T=T, fit_column=fit_column,
                                              use_OCO_table=use_OCO_table, adjust_ch4_broadening=adjust_ch4_broadening)




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


        
        shift = haskey(x, "shift") ? x["shift"] : 1.0
                spectra_grid = shift .* spectra[x_fields[1]].grid
        len_spectra = length(spectra_grid)
        len_measurement = length(measurement.grid)
                
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
        sza = haskey(inversion_setup, "sza") ? inversion_setup["sza"] : 0.0
        
        transmission = calculate_transmission(x, spectra, p, T, input_is_column=inversion_setup["fit_column"], sza=sza)
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



function generate_lhr_model(xₐ::AbstractDict, measurement::AbstractMeasurement, spectra::AbstractDict, spectral_windows, inversion_setup::AbstractDict)
    x_fields = collect(keys(xₐ))
    num_levels = length(xₐ[x_fields[1]])
    
    
    function f(x)

        if typeof(x) <: Array
            FT = eltype(x)
            x = assemble_state_vector!(x, x_fields, num_levels, inversion_setup)
        else
            FT = eltype(x[x_fields[1]])
        end
                
        shift = haskey(x, "shift") ? x["shift"] : 1.0
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
	degree_ind = 1
	out::Vector{FT} = []

        for (i, spectral_window) in enumerate(spectral_windows)

            measurement_grid = find_measurement_grid(spectral_window, measurement.grid)
	    δν = measurement_grid[2] - measurement_grid[1]
            spectral_grid = spectral_window[1]-1.0:δν/2:spectral_window[end]+1.0
            len_spectra = length(spectral_grid)
            len_measurement = length(measurement_grid)
                poly_degree = inversion_setup["poly_degree"][i]
		shape_params = x["shape_parameters"][degree_ind:degree_ind+poly_degree-1]
                degree_ine = degree_ind + poly_degree
		

            transmission = calculate_transmission(x, spectra, p, T, spectral_grid, input_is_column=inversion_setup["fit_column"])
            intensity = apply_instrument(spectral_grid, transmission, measurement_grid)

            # calculate lgendre polynomial coefficients and fit baseline
            if inversion_setup["linear"]
                intensity = log.(intensity)
                intensity .+= calc_polynomial_term(poly_degree, shape_params, len_measurement)
            else
                intensity .*= calc_polynomial_term(poly_degree, shape_params, len_measurement)
            end
	    out = append!(out, intensity)
        end # for loop over windows
            # append intensity into a new array here 
        return out
    end
    return f
end

