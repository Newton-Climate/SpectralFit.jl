


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
                                p::Real=1000, T::Real=290, use_OCO_table=false, adjust_ch4_broadening=false, fit_column::Bool,
                                σ_cache=nothing)

    FT = dicttype(x)
    τ = zeros(FT, grid_length)
    H2O = haskey(x, "H2O") ? x["H2O"] : 0.0
    vcd_wet = calc_vcd(p, T, pathlength)

    # vmr_H2O is the dry-air mole fraction in both modes; column_per_unit converts a state entry to a column
    if fit_column # state entries are columns (molecules/cm²)
        vmr_H2O = H2O / (vcd_wet - H2O)
        column_per_unit = 1.0
    else # state entries are dry-air mole fractions
        vmr_H2O = H2O
        column_per_unit = vcd_wet / (1 + vmr_H2O) # dry-air column
    end
    
   
    for molecule in keys(spectra)
 
        if spectra[molecule].molecule_num == 6 && adjust_ch4_broadening
        println("adjusting CH4 pressure")
            p_adjusted = p*(1+0.34*vmr_H2O)
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p_adjusted, T)

        elseif spectra[molecule].molecule_num == 2 && use_OCO_table
            println("using CO2 tables")
            σ = spectra[molecule].model.itp(spectra[molecule].grid, p, T, vmr_H2O)

        elseif σ_cache === nothing
            σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
        else
            # p and T are fixed, so the cross-section is the same on every call
            σ = get!(() -> absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T), σ_cache, molecule)
        end

        τ .+= (x[molecule] * column_per_unit) .* σ
    end
    return exp.(-τ)
end

function calculate_transmission(x::AbstractDict, spectra::AbstractDict, grid_length::Int; p::Real=1000, T::Real=300)

    FT = dicttype(x)
    τ = zeros(FT, grid_length)
    
    for molecule in keys(spectra)
        σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
        τ .+= x[molecule] .* σ
    end
    return exp.(-τ)
end


"""
Optical-depth operator of each molecule in a layered atmosphere: a matrix (spectral grid × layer)
whose column i is the cross-section at (p[i], T[i]) times the layer's column and the air-mass factor.
The profile's optical depth is then τ = Σₘ A[m] * x[m].
"""
function layer_optical_depth_operators(spectra::AbstractDict, p::Vector{<:Real}, T::Vector{<:Real}; input_is_column=false, sza=0.0)
    vcd = input_is_column ? ones(length(p)) : make_vcd_profile(p, T)
    amf = 1.0/cosd(sza)
    return OrderedDict(molecule => reduce(hcat, [(vcd[i]*amf) .* absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p[i], T[i])
                                                 for i in eachindex(p)])
                       for molecule in keys(spectra))
end

"""transmission of a layered atmosphere from the operators of layer_optical_depth_operators"""
function calculate_transmission(xₐ::AbstractDict, A::AbstractDict{<:Any,<:AbstractMatrix})
    τ = sum(A[molecule] * xₐ[molecule] for molecule in keys(A))
    return exp.(-τ)
end

"""calculate transmission in a profile with multiple layers"""
function calculate_transmission(xₐ::AbstractDict, spectra::AbstractDict, p::Vector{<:Real}, T::Vector{<:Real}; input_is_column=false, sza=0.0)
    return calculate_transmission(xₐ, layer_optical_depth_operators(spectra, p, T, input_is_column=input_is_column, sza=sza))
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
            τ .+= (vcd[i]*xₐ[molecule][i]) .* σ
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
    itp = interpolate(input_spectra, BSpline(Cubic(Line(OnGrid()))))
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



"""
Holds cross-sections computed for one set of atmospheric conditions. `cached!` returns the
stored value while the key (copied p, T, ...) and the spectra's models are unchanged, and
recomputes it otherwise, so a measurement or spectra edited after the model was built is
still handled correctly.
"""
mutable struct CrossSectionCache
    key::Any
    models::Vector{Any}
    value::Any
    CrossSectionCache() = new(nothing, Any[], nothing)
end

function cached!(compute, cache::CrossSectionCache, key, spectra::AbstractDict)
    models = Any[spectra[m].model for m in keys(spectra)]
    if cache.value === nothing || !isequal(cache.key, key) || length(models) != length(cache.models) ||
       !all(cache.models .=== models)
        cache.value = compute()
        cache.key = deepcopy(key)
        cache.models = models
    end
    return cache.value
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
    

    # required: column vs mole-fraction state changes the physics, so there is no silent default
    haskey(inversion_setup, "fit_column") || error("inversion_setup must set \"fit_column\" (true: state in molecules/cm², false: dry-air mole fractions)")

    # Save the labelled fields in the state vector
    x_fields = collect(keys(x₀))
    # cross-sections are reused between calls while p and T are not fitted
    cache = CrossSectionCache()

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
        fit_column = inversion_setup["fit_column"]



                
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature

        fixed_pT = !haskey(x, "pressure") && !haskey(x, "temperature")
        σ_cache = fixed_pT ? cached!(() -> Dict{String,Vector{Float64}}(), cache, (p, T), spectra) : nothing

        # apply Beer's Law
        transmission = calculate_transmission(x, spectra, measurement.pathlength, len_spectra,
                                              p=p, T=T, fit_column=fit_column, σ_cache=σ_cache,
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
    # per-layer cross-sections are computed once while p and T are not fitted
    cache = CrossSectionCache()

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
        
        input_is_column = inversion_setup["fit_column"]
        if !haskey(x, "pressure") && !haskey(x, "temperature")
            A = cached!(() -> layer_optical_depth_operators(spectra, p, T, input_is_column=input_is_column, sza=sza),
                        cache, (p, T, input_is_column, sza), spectra)
            transmission = calculate_transmission(x, A)
        else
            transmission = calculate_transmission(x, spectra, p, T, input_is_column=input_is_column, sza=sza)
        end
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
                degree_ind = degree_ind + poly_degree
		

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

