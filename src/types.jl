
"""
stores the metadata of hitran line-lists and coefficients in type HitranTable

"""
Base.@kwdef struct MolecularMetaData
    molecule::String
    filename::String
    molecule_num::Int
    isotope_num::Int
    grid::Union{AbstractRange, AbstractArray}
    model::AbstractCrossSectionModel
end


"""type to store cross-sections, pressure, temperature, and line-list parameters"""
Base.@kwdef mutable struct Molecule{FT}
    cross_sections::Array{<:Real,1}
    grid::Union{AbstractRange, Array{FT,1}}
    p::FT
    T::FT
    model::AbstractCrossSectionModel
end

""" Save the retrieval results"""
abstract type AbstractResults end


    """-type for storing results from spectral fit
- used later for saving into NetCDF files"""
Base.@kwdef mutable struct InversionResults{FT} <: AbstractResults
    timestamp::DateTime
    machine_time::FT
    x::Union{AbstractDict{String, Union{FT, Vector{FT}}}, AbstractDict{String, Vector{FT}}}
    measurement::Array{FT,1}
    model::Array{FT,1}
    χ²::FT
    S::Array{FT,2}
    grid::Array{FT,1}
    K::Array{FT,2}
    Sₑ⁻¹::AbstractArray{FT}
    Sₐ⁻¹::AbstractArray{FT}
end


Base.@kwdef mutable struct FailedInversion{FT} <: AbstractResults
    timestamp::DateTime
    machine_time::FT
    x::Union{AbstractDict{String, Union{FT, Vector{FT}}}, AbstractDict{String, Vector{FT}}}
    measurement::Array{FT,1}
    grid::Vector{FT}
end

    

abstract type AbstractDataset end


Base.@kwdef mutable struct FrequencyCombDataset{FT} <: AbstractDataset
    filename::String
    intensity::Array{FT,2}
    grid::Vector{FT}
    temperature::Vector{FT}
    pressure::Vector{FT}
    time::Vector{DateTime}
    pathlength::FT
    num_averaged_measurements::Vector{Int}
    averaging_window::Dates.Period
    machine_time::Vector{FT}
    σ²::Vector{FT}
    vcd::Vector{FT}
end


abstract type AbstractMeasurement end

Base.@kwdef mutable struct FrequencyCombMeasurement{FT} <: AbstractMeasurement
    intensity::Vector{FT}
    grid::Vector{FT}
    temperature::Union{FT, Vector{FT}}
    pressure::Union{FT, Vector{FT}}
    time::Dates.DateTime
    pathlength::FT
    vcd::Union{FT, Array{FT,1}}
    num_averaged_measurements::Int
    averaging_window::Any
    machine_time::FT
    σ²::FT
end
    
Base.@kwdef struct SimpleInterpolationModel{FT} <: AbstractCrossSectionModel
    itp
    mol::Int
    iso::Int
    ν_grid::Union{Array{FT,1}, UnitRange{FT}}
    p_grid::Union{Array{FT,1}, AbstractRange{FT}}
    T_grid::Union{Array{FT,1}, AbstractRange{FT}}
end
