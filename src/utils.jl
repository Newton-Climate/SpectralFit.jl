using RecursiveArrayTools, Polynomials

"""Finds the indexes given values ν_min:ν_max"""
function find_indexes(ν_min::Real, ν_max::Real, ν_grid::Array{Float64,1})
    
    indexes = findall(x-> ν_min<x<ν_max, ν_grid)
    return indexes
end #function find_indexes

"""Calculate the vertical column density given pressure and temperature"""
function vcd_pressure(δp::Real, T::Real, vmr_H₂O::Real)
    dry_mass = 28.9647e-3  /Nₐ  # in kg/molec, weighted average for N2 and O2
    wet_mass = 18.01528e-3 /Nₐ  # just H2O
    ratio = dry_mass/wet_mass
    vmr_dry = 1 - vmr_H₂O
    M  = vmr_dry * dry_mass + vmr_H₂O * wet_mass
    vcd_dry = 100.0*vmr_dry*δp/(M*g₀*100.0^2)   #includes m2->cm2
    #vcd_H₂O = vmr_H₂O*δp/(M*g₀*100^2)
    return vcd_dry #+ vcd_H₂O
end

"""Calculate the vertical column density given humidity, pressure, temperature, and layer thickness"""
function calc_vcd(p::Real, T::Real, δz::Float64, VMR_H₂O::Real)
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end # function calc_vcd

"""Calculate the vertical column density given pressure, temperature, and layer thickness"""
function calc_vcd(p::Real, T::Real, δz::Float64)
    VMR_H₂O = 0
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end #function calc_vcd

"""Calculate the half-pressure levels given a pressure profile"""
function half_pressure_levels(p::Array{FT,1}) where FT <: Real
    half_levels = zeros(FT, length(p)+1)

    p₀ = p[1]
       
    for i=2:length(p)
        half_levels[i] = (p[i] + p[i-1])/2
    end

    # take care of top and bottom boundaries
    # replace with the δp nearest to top/bottom
    half_levels[1] = p₀ - (half_levels[2]-p₀)
    half_levels[end] = p[end] + (p[end]-half_levels[end-1])


    return abs.(half_levels[2:end] - half_levels[1:end-1])
        

end

"""construct vcd profiles by layer, given pressure,  temperature, and humidity"""
function make_vcd_profile(p::Array{<:Real,1}, T::Array{<:Real,1}; vmr_H₂O=nothing)
    
    if vmr_H₂O == nothing
        vmr_H₂O = zeros(length(p))
    end
    if p[end] == p[1]
        δp = p
        T = half_pressure_levels(T)
    else
            
        δp = half_pressure_levels(p)
    end
    
    input_variables = zip(δp,T,vmr_H₂O)
    vcd = map(x -> vcd_pressure(x[1], x[2], x[3]), input_variables)
    return vcd
end

function dicttype(x::AbstractDict)
    k = collect(keys(x))
    FT = eltype(x[k[1]])
    return FT
end


"""Convert a state vector{Dict} to an Array"""
function assemble_state_vector!(x::AbstractDict)

    key = collect(keys(x))
    FT = eltype(x[key[1]])
    out::Vector{FT} = []
    for key in keys(x)
        out = append!(out, x[key])
    end
    return out
end #function assemble_state_vector!

"""Convert the state vecotr{Array} to a Dict"""
function assemble_state_vector!(x::Vector{FT}, key_vector, inversion_setup::AbstractDict) where FT <: Real

    out::OrderedDict{String, Union{FT, Vector{FT}}} = OrderedDict([key_vector[i] => x[i] for i=1:length(key_vector)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end #function assemble_state_vector!


"""Convert the state vecotr{Array} to a Dict"""
function assemble_state_vector!(x::Array{FT,1}, fields::AbstractArray, num_levels::Integer, inversion_setup::AbstractDict) where FT<:Real
    out = OrderedDict{String, Vector{FT}}(fields[i] => x[1+(i-1)*num_levels : i*num_levels] for i=1:length(fields)-1)
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end


"""calculates the legendre polynomial over domain x::Vector of degree max::Integer"""
function compute_legendre_poly(x::Array{<:Real,1}, nmax::Integer)
    
    @assert nmax > 1
    #@assert size(P) == (nmax,length(x))
    P⁰ = zeros(nmax,length(x));
   
    # 0th Legendre polynomial, a constant
    P⁰[1,:] .= 1.0;

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


function prior_shape_params(dataset::AbstractDataset,
                            inversion_setup::AbstractDict)
    return [maximum(dataset.intensity); zeros(inversion_setup["poly_degree"]-1)]
end

    
"""Calculate the gain matrix from InversionResults fields"""
function calc_gain_matrix(inversion_results::InversionResults)
    G = inv(inversion_results.K'*inversion_results.Sₑ⁻¹*inversion_results.K + inversion_results.Sₐ⁻¹)*inversion_results.K'*inversion_results.Sₑ⁻¹
    return G
end

function calc_χ²(result::InversionResults)
    if typeof(result.x) <: AbstractDict
        x = assemble_state_vector!(result.x)
    else
        x=result.x
    end
    
    χ² = (result.model - result.measurement)'* result.Sₑ⁻¹ * (result.model - result.measurement) / (length(result.model)-length(x))
    return χ²
end

function calc_DCS_noise(data::FrequencyCombDataset)

    # select spectrally flat region 
    ν_min, ν_max = 6255.1, 6255.4

    # find wavelength indices:
    ind = intersect(findall(data.grid .> ν_min), findall(data.grid .< ν_max))

    signal_total = mean(data.intensity, dims=2);

    # Subset data for spectral range
    data_subset = data.intensity[:,ind]
    grid_subset = data.grid[ind] .- mean(data.grid[ind])

    # Number of spectra in there:
    n_spec = size(data_subset,1)

    # loop over all individual spectra:
    mod = similar(data_subset)
    for i=1:n_spec
        # take that time step
        flat_region =  data_subset[i,:]

        # fit out the baseline with a 3rd degree polynomial
        fitted = fit(grid_subset, flat_region, 3)

        # evaluate the trend given the polynomial coefficients
        mod[i,:] = fitted.(grid_subset)
    end

    # Mean residual (has little impact here)
    mm = mean(mod' .- data_subset', dims=2)
    # Standard deviation from fit (mean residual removed):
    sm = std(mod' .- data_subset' .- mm, dims=1)

    # Fit noise model (linear with offset):
    slope_noise_fit = sqrt.(signal_total[:,1]) \ sm[1,:]

    # This will now give you the total noise, i.e. for an individual (single) sounding, the noise 1sigma is just Se = slope_noise_fit * sqrt(signal_total[:,1])
    σ = mean(slope_noise_fit * sqrt.(signal_total[:,1]))
    return σ
end


function calc_DCS_noise(grid::Array{Float64,1}, intensity::Array{Float64,2})

    signal_total = mean(intensity, dims=2);

    # find wavelength indices over flat region :
        ν_min, ν_max = 6255.1, 6255.4
    ind = intersect(findall(grid .> ν_min), findall(grid .< ν_max))

    # Subset data for spectral range
    data_subset = intensity[:,ind]
    grid_subset = grid[ind] .- mean(grid[ind])

    # Number of spectra in there:
    n_spec = size(data_subset,1)

    # loop over all individual spectra:
    mod = similar(data_subset)
    for i=1:n_spec
        # take that time step
        flat_region =  data_subset[i,:]

        # fit out the baseline with a 3rd degree polynomial
        fitted = fit(grid_subset, flat_region, 3)

        # evaluate the trend given the polynomial coefficients
        mod[i,:] = fitted.(grid_subset)
    end

    # Mean residual (has little impact here)
    mm = mean(mod' .- data_subset', dims=2)
    # Standard deviation from fit (mean residual removed):
    sm = std(mod' .- data_subset' .- mm, dims=1)

    # Fit noise model (linear with offset):
    slope_noise_fit = sqrt.(signal_total[:,1]) \ sm[1,:]

    # This will now give you the total noise, i.e. for an individual (single) sounding, the noise 1sigma is just Se = slope_noise_fit * sqrt(signal_total[:,1])
    σ = slope_noise_fit * sqrt.(signal_total[:,1])
    return σ.^2
end

"""Find the indexes of the spectral windows being masked"""
function find_mask(spectral_grid::Vector{<:Real}, # spectral grid 
                   masked_windows::Array{<:Real,2}) # spectral windows to mask in the form [ν₁, ν₂]
    
    # Find all indices in the grid that is within the subwindows:
    all = [findall(i -> (i>masked_windows[j,1])&(i<masked_windows[j,2]), spectral_grid) for j=1:size(masked_windows,1)];
    
    # Return all indices:
    vcat(all...)
end
    
