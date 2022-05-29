
using Interpolations, Statistics
using ForwardDiff
using LinearAlgebra, Statistics 

include("spectroscopy.jl")
include("read_data.jl")

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
function calculate_transmission!(τ::Vector{<:Real}, x::AbstractDict, pathlength::Real, spectra::AbstractDict; p::Real=1000, T::Real=290)

    #σ = similar(τ)
    vcd = calc_vcd(p, T, pathlength, x["H2O"])
    
    for molecule in keys(spectra)
        σ = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
    τ .+= @. vcd * x[molecule] * σ
    end
    return exp.(-τ)
end

function calculate_transmission(x::AbstractDict, spectra::AbstractDict; p::Real=1000, T::Real=300)

    k = collect(keys(spectra))
        FT = eltype(x[k[1]].cross_sections)
    τ = zeros(FT, size(spectra[k[1]].cross_sections));
    σ = similar(τ)
    
    for molecule in keys(spectra)
        σ[:] = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p, T)
    τ += @. x[molecule] * σ
    end
    return exp.(-τ)
end

function calculate_transmission!(τ::Vector{<:Real},
                                 x::AbstractDict,
                                 spectra::AbstractDict;
                                 p::Real=1000, T::Real=300)
    
    σ = similar(τ)
    for molecule in keys(spectra)
        σ[:] = absorption_cross_section(spectra[molecule].model, spectra[molecule].grid, p,T)
    τ += @. x[molecule] * σ
    end
    return exp.(-τ)
end


"""calculate transmission in a profile with multiple layers"""
function calculate_transmission(xₐ::AbstractDict, spectra::Array{<:AbstractDict,1}, vcd::Array{<:Real,1})
   
    n_levels = length(vcd)
    k = collect(keys(spectra[1]))
        FT = eltype(x[k[1]].cross_sections)
    τ = zeros(FT, size(spectra[1][k[1]].grid))
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
function apply_instrument( input_spectral_grid::AbstractArray{<:Real},
                               input_spectra::AbstractArray{<:Real},
                           output_spectral_grid::AbstractArray{<:Real})
    
    #itp = interpolate(input_spectra, BSpline(Quadratic(Line(OnGrid()))))
    itp = interpolate(input_spectra, BSpline(Linear()))
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
    x₀_fields = collect(keys(x₀))
    
    function f(x)
        # convert the state vector to a dict with labelled fields
        if typeof(x) <: AbstractArray
            FT = eltype(x)
            x = assemble_state_vector!(x, x₀_fields, inversion_setup)
        else
            FT = eltype(x[x₀_fields[1]])
        end

        # the output vector
        spectra_grid = spectra[x₀_fields[1]].grid
        len_spectra = length(spectra_grid)
        len_measurement = length(measurement.grid)
        
        transmission = zeros(FT, len_spectra)
        intensity = zeros(FT, len_measurement)
        polynomial_term = zeros(FT, len_measurement)
        
        p = haskey(x, "pressure") ? x["pressure"] : measurement.pressure
        T = haskey(x, "temperature") ? x["temperature"] : measurement.temperature
        
        #for the OCO line-list for CO₂
        if inversion_setup["use_OCO"] && spectra["CO2"].grid[1] >= 6140 && spectra["CO2"].grid[end] <= 6300

            spectra["CO2"].cross_sections = OCO_interp(spectra["CO2"].grid, x["temperature"], x["pressure"])
        end
        
        # apply Beer's Law
        if haskey(inversion_setup, "fit_column") && inversion_setup["fit_column"] == true
            transmission = calculate_transmission!(transmission, x, spectra, p=p, T=T)
        else
            transmission = calculate_transmission!(transmission, x, measurement.pathlength, spectra, p=p, T=T)
        end

        # down-sample to instrument grid
        intensity[:] = apply_instrument(spectra_grid, transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline
        intensity .*= calc_polynomial_term(inversion_setup["poly_degree"], x["shape_parameters"], len_measurement)


        #intensity .*= polynomial_term
        return intensity
    end
    return f
end


"""Generate a foreward model that calculates the transmission through multiple layers of the atmosphere"""
function generate_profile_model(xₐ::AbstractDict, measurement::AbstractMeasurement, spectra::Array{OrderedDict,1}, inversion_setup::AbstractDict)
    x_fields = collect(keys(xₐ))
    num_levels = length(xₐ[x_fields[1]])
    
    
    function f(x::AbstractArray)

        FT = eltype(x)
        intensity = zeros(FT, length(measurement.grid))

        if typeof(x) <: Array
            x = assemble_state_vector!(x, x_fields, num_levels, inversion_setup)
        end
        if inversion_setup["fit_pressure"]
            p, T = x["pressure"], x["temperature"]
                    spectra = construct_spectra_by_layer!(spectra, p=p, T=T)
        end
        
        transmission = calculate_transmission(x, spectra, measurement.vcd)
        transmission = apply_instrument(collect(spectra[1][x_fields[1]].grid), transmission, measurement.grid)

        # calculate lgendre polynomial coefficients and fit baseline 
        shape_parameters = x["shape_parameters"]
        polynomial_term = calc_polynomial_term(inversion_setup["poly_degree"], shape_parameters, length(transmission))'
        intensity[:] = transmission .* polynomial_term
        return intensity
    end
    return f
end

xₐ = OrderedDict{String,Union{Float64, Vector{Float64}}}("H2O" => 0.01,
    "CH4" => 2000e-9,
                  "CO2" => 400e-6,
                  "pressure" => 1000.0,
                  "temperature" => 300.0,
                  "shape_parameters" => [maximum(measurement.intensity); ones(inversion_setup["poly_degree"]-1)])

# just testing the fit itself
f = generate_forward_model(xₐ, measurement, spec, inversion_setup);
@time out = f(xₐ)
#@time out = f(xₐ)
x = assemble_state_vector!(xₐ)
#@time f(x)
#@time f(x)
#
cfg10 = ForwardDiff.JacobianConfig(f,x, ForwardDiff.Chunk{30}())
x = x
println("chunk = 1")

@time k = ForwardDiff.jacobian(f,x, cfg10)
