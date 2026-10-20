# Parameter slots: a quantity that is either fixed or read from the state vector, decided
# once when a forward model is built (section 3.2 of docs/architecture_plan.md).

"""Supertype of parameter slots: `Fixed` or `FromState`."""
abstract type AbstractParameter end

"""
    Fixed(value)

A parameter slot holding a known value. `resolve(Fixed(3.0), x) == 3.0`.
"""
struct Fixed{T} <: AbstractParameter
    value::T
end

"""
    FromState{Path}()

A parameter slot read from the state vector. `Path` is a `Symbol` (`:pressure_hPa`) or a tuple
of symbols for a nested entry (`(:instrument, :stretch)`). Because the path is a type
parameter, `resolve` compiles to a field access.
"""
struct FromState{Path} <: AbstractParameter end

"""
    resolve(parameter, x)

Value of a parameter slot for state `x`: the stored value of a `Fixed`, or the state entry of a
`FromState`.

```julia
x = ComponentVector(a = 2.0, instrument = (stretch = 1.0,))
resolve(Fixed(3.0), x)                         # 3.0
resolve(FromState{:a}(), x)                    # 2.0
resolve(FromState{(:instrument, :stretch)}(), x)  # 1.0
```
"""
@inline resolve(p::Fixed, x) = p.value
@inline resolve(::FromState{Path}, x) where {Path} = get_state_path(x, Path)

@inline get_state_path(x, name::Symbol) = getproperty(x, name)
@inline get_state_path(x, path::Tuple{Symbol}) = getproperty(x, path[1])
@inline get_state_path(x, path::Tuple{Symbol,Vararg{Symbol}}) =
    get_state_path(getproperty(x, path[1]), Base.tail(path))

"""
    as_parameter(value)

Wrap a plain value in `Fixed`; return a parameter slot unchanged. Used by constructors that
accept either.
"""
as_parameter(p::AbstractParameter) = p
as_parameter(value) = Fixed(value)

"""
    has_state_path(x, path) -> Bool

Whether the state `x` (a `ComponentVector`) has the entry `path` (a `Symbol` or tuple of
symbols).
"""
has_state_path(x, name::Symbol) = has_state_path(x, (name,))
function has_state_path(x, path::Tuple)
    isempty(path) && return true
    x isa ComponentArray || return false
    first(path) in propertynames(x) || return false
    return has_state_path(getproperty(x, first(path)), Base.tail(path))
end

"""
    bind_parameter(parameter, path, x_prior)

Turn a `Fixed` slot into `FromState{path}()` when `x_prior` has the entry `path`; keep it
otherwise. A `FromState` slot whose path is missing from `x_prior` is an error.
"""
bind_parameter(p::Fixed, path, x_prior) =
    has_state_path(x_prior, path) ? FromState{path}() : p
function bind_parameter(p::FromState{Path}, path, x_prior) where {Path}
    has_state_path(x_prior, Path) ||
        throw(ArgumentError("parameter slot reads state entry $(Path), which is not in the state"))
    return p
end

"""
    bind_parameters(component, x_prior)

Return `component` with each of its parameter slots bound against the state names in
`x_prior` (see `bind_parameter`). Components with slots add a method; the fallback returns
the component unchanged. Called by the forward-model builder.
"""
bind_parameters(component, x_prior) = component

export AbstractParameter, Fixed, FromState, resolve, as_parameter
export has_state_path, get_state_path, bind_parameter, bind_parameters
