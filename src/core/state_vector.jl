# State-vector helpers. The state is a ComponentArrays.ComponentVector (section 3.3).

"""
    prior_precision_from_sigma(sigma::ComponentVector[, x_prior::ComponentVector])

Diagonal prior precision (the inverse of Sa) from prior standard deviations. When `x_prior`
is given, the axes of `sigma` must equal those of `x_prior`. Every sigma must be positive.

```julia
sigma = ComponentVector(CH4 = 1e17, baseline = (window_1 = fill(1e3, 5),))
prior_precision_from_sigma(sigma)   # Diagonal(1 ./ sigma .^ 2)
```
"""
function prior_precision_from_sigma(sigma::ComponentVector)
    all(s -> isfinite(s) && s > 0, sigma) ||
        throw(ArgumentError("prior_sigma: every standard deviation must be positive and finite"))
    return Diagonal(1 ./ collect(sigma) .^ 2)
end

function prior_precision_from_sigma(sigma::ComponentVector, x_prior::ComponentVector)
    getaxes(sigma) == getaxes(x_prior) ||
        throw(ArgumentError("prior_sigma: axes $(getaxes(sigma)) do not match x_prior axes $(getaxes(x_prior))"))
    return prior_precision_from_sigma(sigma)
end

"""
    column_to_vmr(x, total_column; gases, h2o = :H2O)

Convert gas entries `gases` (tuple of symbols) of a homogeneous-path state from columns
(molecules/cm2) to dry-air mole fractions (mol/mol). `total_column` is the wet-air column of
the path from the ideal gas law. The dry-air column is `total_column - N_H2O`; H2O becomes its
dry mole fraction `N_H2O / (total_column - N_H2O)`. Without an `h2o` entry the air is dry.
Non-gas entries are copied unchanged. Inverse of `vmr_to_column`.
"""
function column_to_vmr(x::ComponentVector, total_column; gases, h2o::Symbol = :H2O)
    n_h2o = has_state_path(x, h2o) ? getproperty(x, h2o) : zero(total_column)
    dry_column = total_column .- n_h2o
    y = copy(x)
    for gas in gases
        setproperty!(y, gas, getproperty(x, gas) ./ dry_column)
    end
    return y
end

"""
    vmr_to_column(x, total_column; gases, h2o = :H2O)

Convert gas entries of a homogeneous-path state from dry-air mole fractions to columns. The
dry-air column is `total_column / (1 + q_H2O)`. Inverse of `column_to_vmr`.
"""
function vmr_to_column(x::ComponentVector, total_column; gases, h2o::Symbol = :H2O)
    q_h2o = has_state_path(x, h2o) ? getproperty(x, h2o) : zero(total_column)
    dry_column = total_column ./ (1 .+ q_h2o)
    y = copy(x)
    for gas in gases
        setproperty!(y, gas, getproperty(x, gas) .* dry_column)
    end
    return y
end

"""
    state_indices(x::ComponentVector, name::Symbol) -> Vector{Int}

Flat indices of the top-level state entry `name` (all its elements, nested ones included).

```julia
x = ComponentVector(H2O = [1.0, 2.0], CH4 = [3.0, 4.0], baseline = (window_1 = [1.0],))
state_indices(x, :CH4)    # [3, 4]
```
"""
function state_indices(x::ComponentVector, name::Symbol)
    hasproperty(x, name) || throw(ArgumentError("state has no entry $name; entries: $(propertynames(x))"))
    prefix = String(name)
    return findall(l -> l == prefix || startswith(l, prefix * "[") || startswith(l, prefix * "."),
                   Vector{String}(labels(x)))
end

export prior_precision_from_sigma, column_to_vmr, vmr_to_column, state_indices
