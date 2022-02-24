using RecursiveArrayTools, Polynomials

"""Finds the indexes given values ν_min:ν_max"""
function find_indexes(ν_min::Real, ν_max::Real, ν_grid::Array{Float64,1})
    
    a = findlast(x -> x <= ν_min, ν_grid)
    b = findfirst(x -> x >= ν_max, ν_grid)
    indexes = collect(a:b)
    return indexes
end #function find_indexes

"""Calculate the vertical column density given pressure and temperature"""
function vcd_pressure(δp::Real, T::Real, vmr_H₂O::Real)
    δp = δp*100 # convert from mbar to pascals 
    dry_mass = 28.9647e-3  /Nₐ  # in kg/molec, weighted average for N2 and O2
    wet_mass = 18.01528e-3 /Nₐ  # just H2O
    ratio = dry_mass/wet_mass
    vmr_dry = 1 - vmr_H₂O
    M  = vmr_dry * dry_mass + vmr_H₂O * wet_mass
    vcd_dry = vmr_dry*δp/(M*g₀*100.0^2)   #includes m2->cm2
    vcd_H₂O = vmr_H₂O*δp/(M*g₀*100^2)
    return vcd_dry #+ vcd_H₂O
end

"""Calculate the vertical column density given humidity, pressure, temperature, and layer thickness"""
function calc_vcd(p::Float64, T::Float64, δz::Float64, VMR_H₂O::Float64)    
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


"""construct vcd profiles by layer, given pressure,  temperature, and humidity"""
function make_vcd_profile(p::Array{<:Real,1}, T::Array{<:Real,1}; vmr_H₂O=nothing)
    vcd = zeros(length(p))
    
    if vmr_H₂O == nothing
        vmr_H₂O = zeros(length(p))
    end
    
    half_levels = half_pressure_levels(p)
    δp = half_levels[2:end] - half_levels[1:end-1]
    input_variables = zip(δp,T,vmr_H₂O)
    vcd = map(x -> SpectralFits.vcd_pressure(x[1], x[2], x[3]), input_variables)
    return vcd
end

"""Convert a state vector{Dict} to an Array"""
function assemble_state_vector!(x::AbstractDict)
    out::Array{Real,1} = []
    for key in keys(x)
        out = append!(out, x[key])
        end
    return out
end #function assemble_state_vector!

"""Convert the state vecotr{Array} to a Dict"""
function assemble_state_vector!(x::Vector{<:Real}, key_vector::Array{Any,1}, inversion_setup::AbstractDict)
    out::OrderedDict{Any,Any} = OrderedDict([key_vector[i] => x[i] for i=1:length(key_vector)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end #function assemble_state_vector!


"""Convert the state vecotr{Array} to a Dict"""
function assemble_state_vector!(x::Array{<:Real,1}, fields::Array{Any,1}, num_levels::Integer, inversion_setup::AbstractDict)
    out::OrderedDict{Any, Array{<:Real,1}} = OrderedDict([fields[i] => x[1+(i-1)*num_levels : i*num_levels] for i=1:length(fields)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end


"""calculates the legendre polynomial over domain x::Vector of degree max::Integer"""
function compute_legendre_poly(x::Array{<:Real,1}, nmax::Integer)
    
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

"""Calculate the gain matrix from InversionResults fields"""
function calc_gain_matrix(inversion_results::InversionResults)
    G = inv(inversion_results.K'*inversion_results.Sₑ*inversion_results.K + inversion_results.Sₐ)*inversion_results.K'*inversion_results.Sₑ
    return G
end

function calc_χ²(result::InversionResults)
    if typeof(result.x) <: AbstractDict
        x = assemble_state_vector!(result.x)
    else
        x=result.x
    end
    
    χ² = (result.model - result.measurement)'* result.Sₑ * (result.model - result.measurement) / (length(result.model)-length(x))
    return χ²
end

"""convert an array of arrays to an array"""
function list2array(array_in::Array)
    outer_dim, inner_dim = length(array_in), length(array_in[1])
    data_type = typeof(array_in[1][1])

    # allocate memory
    array_out = Array{data_type}(undef, (outer_dim, inner_dim))
    v = VectorOfArray(array_in)
    array_out = convert(Array, v)
    return array_out'
end

"""convert an array of arrays to an array"""
function list2array!(array_out::Array, array_in::Array)
    v = VectorOfArray(array_in)
    array_out = convert(Array, v)
    return array_out'
end

function calc_DCS_noise(data::Dataset)

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
    return σ
end


    
