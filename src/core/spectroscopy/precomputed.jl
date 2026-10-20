# Cross sections already evaluated on the fine grid (section 3.4). Kept in its own file
# because the atmosphere code and the builder both depend on it.

"""
    Precomputed(sigma::AbstractMatrix)

Cross sections in cm2/molecule already evaluated on the fine grid, one column per atmospheric
layer. Depends on no conditions. The builder creates it with `precompute` when none of the
conditions an absorber depends on are retrieved.
"""
struct Precomputed{M<:AbstractMatrix{Float64}} <: AbstractCrossSectionModel
    sigma::M
end

depends_on(::Precomputed) = ()
spectral_range_cm(::Precomputed) = (-Inf, Inf)
layer_cross_section(model::Precomputed, wavenumber_cm, conditions, layer) =
    view(model.sigma, :, layer)
function cross_section(model::Precomputed, wavenumber_cm, conditions)
    size(model.sigma, 2) == 1 ||
        throw(ArgumentError("Precomputed holds $(size(model.sigma, 2)) layers; use layer_cross_section"))
    return view(model.sigma, :, 1)
end

"""
    precompute(model, wavenumber_cm, layer_conditions::AbstractVector{<:GasConditions})

Evaluate `model` once per layer and store the result as `Precomputed`.
"""
function precompute(model::AbstractCrossSectionModel, wavenumber_cm,
                    conditions::AbstractVector{<:GasConditions})
    sigma = Matrix{Float64}(undef, length(wavenumber_cm), length(conditions))
    for (layer, c) in enumerate(conditions)
        sigma[:, layer] .= cross_section(model, wavenumber_cm, c)
    end
    return Precomputed(sigma)
end
precompute(model::Precomputed, wavenumber_cm, conditions::AbstractVector{<:GasConditions}) = model

export Precomputed, precompute
