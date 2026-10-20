# Shared fixtures for the SpectralFits test suite. Included once by runtests.jl and by
# run_one.jl; test files may assume everything here is defined.
#
# data/hit20_subset.par is a small slice of HITRAN 2020 (H2O, 12CH4, 12CO2) covering
# 6076-6084 and 6224-6232 cm-1, so the tests run without the full line lists.

if !@isdefined(SPECTRALFITS_TEST_HELPERS_LOADED)
const SPECTRALFITS_TEST_HELPERS_LOADED = true

using Test
using SpectralFits
using SpectralFits.CoreLayer
using ComponentArrays
using Dates, LinearAlgebra, Statistics, Random, SparseArrays
import ForwardDiff

const SF = SpectralFits
const FIXTURE_PAR = joinpath(@__DIR__, "data", "hit20_subset.par")

"""
    AnalyticLorentzLine(center_cm, strength, hwhm_cm_per_hPa)

Test-only cross-section model: one pressure-broadened Lorentz line with line strength
`strength` (cm/molecule, the integral of the cross section over wavenumber) and half width at
half maximum `hwhm_cm_per_hPa * pressure_hPa`. It shows that a new model needs only the three
interface methods.
"""
struct AnalyticLorentzLine <: AbstractCrossSectionModel
    center_cm::Float64
    strength::Float64
    hwhm_cm_per_hPa::Float64
end
function SpectralFits.CoreLayer.cross_section(m::AnalyticLorentzLine, wavenumber_cm, c::GasConditions)
    gamma = m.hwhm_cm_per_hPa * c.pressure_hPa
    return @. m.strength * (gamma / pi) / ((wavenumber_cm - m.center_cm)^2 + gamma^2)
end
SpectralFits.CoreLayer.depends_on(::AnalyticLorentzLine) = (:pressure,)
SpectralFits.CoreLayer.spectral_range_cm(::AnalyticLorentzLine) = (-Inf, Inf)

"""Central finite-difference Jacobian of `f` at `x`, relative step `rel_step`."""
function finite_difference_jacobian(f, x::AbstractVector; rel_step = 1e-6)
    y0 = f(x)
    J = zeros(length(y0), length(x))
    for j in eachindex(x)
        h = rel_step * max(abs(x[j]), 1.0)
        xp = copy(x); xp[j] += h
        xm = copy(x); xm[j] -= h
        J[:, j] = (f(xp) - f(xm)) / (2h)
    end
    return J
end

const FIXTURE_MOLECULES = (H2O = 1, CO2 = 2, CH4 = 6)
const FIXTURE_MODEL_CACHE = Dict{Any,Any}()

"""
    fixture_voigt(gas::Symbol, range_cm) -> HitranVoigt

Line-by-line model of `gas` (:H2O, :CO2 or :CH4, isotope 1) from the HITRAN subset over
`range_cm`, built once per test run with the v0.1 options (wing cutoff 40 cm-1, lines inside
the range only).
"""
fixture_voigt(gas::Symbol, range_cm) = get!(FIXTURE_MODEL_CACHE, (gas, range_cm)) do
    SpectralFits.IOLayer.load_cross_section_model(HitranVoigt, FIXTURE_PAR;
        molecule = FIXTURE_MOLECULES[gas], isotope = 1, range_cm = range_cm)
end

end # if !@isdefined
