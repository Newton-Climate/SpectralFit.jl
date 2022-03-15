using Dates
using ForwardDiff:Dual

    """stores the metadata of hitran line-lists and coefficients in type HitranTable"""
struct MolecularMetaData
    molecule::String
    filename::String
    molecule_num::Integer
    isotope_num::Integer
    ν_grid::AbstractRange
    hitran_table
    model
end


"""type to store cross-sections, pressure, temperature, and line-list parameters"""
mutable struct Molecule{FT}
   
    cross_sections::AbstractArray
    grid::Union{AbstractRange{FT}, Array{FT,1}}
    p::FT
    T::FT
    hitran_table::HitranTable{FT}
    model::HitranModel
end

    """Type to store molecules"""
mutable struct Spectra
    H₂O::Molecule
    CH₄::Molecule
    CO₂::Molecule
    HDO::Molecule
end


    """-type for storing results from spectral fit
- used later for saving into NetCDF files"""
mutable struct InversionResults
    timestamp
    machine_time
    x
    measurement
    model
    χ²
    S
    grid
    K
    Sₑ
    Sₐ
end


abstract type Dataset end

abstract type FrequencyComb <: Dataset end

mutable struct FrequencyCombDataset <: Dataset
    filename::String
    intensity::Array{Float64,2}
    grid::Vector{Float64}
    temperature::Vector{Float64}
    pressure::Vector{Float64}
    time::Vector{Any}
    pathlength::Float64
    timestamp::Array{Float64,1}
    σ²::Vector{Float64}
end 

struct TimeAveragedFrequencyCombDataset <: FrequencyComb
    filename::String
    intensity::Array{Float64,2}
    grid::Vector{Float64}
    temperature::Vector{Float64}
    pressure::Vector{Float64}
    time::Vector{Tuple{DateTime,DateTime}}
    pathlength::Float64
    num_averaged_measurements::Vector{Int64}
    averaging_window::Dates.Period
    timestamp::Array{Float64,1}
    σ²::Vector{Float64}
end


abstract type Measurement end

mutable struct FrequencyCombMeasurement <: Measurement
    intensity::Array{Float64,1}
    grid::Vector{Float64}
    temperature::Union{Float64, Array{Float64,1}}
    pressure::Union{Float64, Array{Float64,1}}
    time::Any
    pathlength::Float64
    vcd::Union{Float64, Array{Float64,1}}
    num_averaged_measurements::Int64
    averaging_window::Any
    machine_time::Float64
    σ²::Float64
end
    
