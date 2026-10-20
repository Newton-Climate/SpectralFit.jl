# Cross-section wrappers and table models (section 3.4 of docs/architecture_plan.md):
# `H2OBroadened`, which evaluates any model at a water-broadened effective pressure, and
# `ABSCOTable`, the OCO-style absorption coefficient tables. `Precomputed` and `precompute`
# live in precomputed.jl.

# ---------------------------------------------------------------- H2OBroadened

"""
    H2OBroadened(inner::AbstractCrossSectionModel, coefficient::Real)

Evaluates `inner` at the effective pressure
`pressure_hPa * (1 + coefficient * h2o_dry_mole_fraction)`, which approximates water-vapour
broadening of the lines by air broadening at a higher pressure. `coefficient` is
dimensionless; 0.34 is the value v0.1 used for CH4 (`adjust_ch4_broadening`). Any absorber
can be wrapped. Depends on what `inner` depends on plus `:h2o`.

```julia
ch4_wet = H2OBroadened(ch4, 0.34)
cross_section(ch4_wet, grid, GasConditions(850.0, 290.0, 0.02))
# equals cross_section(ch4, grid, GasConditions(850.0 * (1 + 0.34 * 0.02), 290.0, 0.02))
```
"""
struct H2OBroadened{M<:AbstractCrossSectionModel} <: AbstractCrossSectionModel
    inner::M
    coefficient::Float64
end
H2OBroadened(inner::AbstractCrossSectionModel, coefficient::Real) =
    H2OBroadened(inner, Float64(coefficient))

"""
    broadened_conditions(model::H2OBroadened, conditions::GasConditions) -> GasConditions

Conditions passed to the inner model: pressure replaced by the effective pressure in hPa.
"""
function broadened_conditions(m::H2OBroadened, c::GasConditions)
    p_effective = c.pressure_hPa * (1 + m.coefficient * c.h2o_dry_mole_fraction)
    return GasConditions(p_effective, c.temperature_K, c.h2o_dry_mole_fraction)
end

cross_section(m::H2OBroadened, wavenumber_cm::AbstractVector, c::GasConditions) =
    cross_section(m.inner, wavenumber_cm, broadened_conditions(m, c))
depends_on(m::H2OBroadened) = union_conditions(depends_on(m.inner), (:h2o,))
spectral_range_cm(m::H2OBroadened) = spectral_range_cm(m.inner)

"""Union of two condition tuples, in the canonical order `(:pressure, :temperature, :h2o)`."""
union_conditions(a::Tuple, b::Tuple) =
    filter(s -> s in a || s in b, (:pressure, :temperature, :h2o))

# ---------------------------------------------------------------- ABSCOTable

"""
    ABSCOTable(sigma::AbstractArray{<:Real,4}, axes::NamedTuple)

OCO-style absorption coefficient table: `sigma[i_nu, i_h2o, i_T, i_p]` in cm2/molecule on

- `axes.wavenumber_cm`: increasing wavenumbers, cm-1;
- `axes.h2o_mole_fraction`: increasing H2O broadener mole fractions, mol/mol of wet air (the
  ABSCO `Broadener_01_VMR` convention);
- `axes.temperature_K`: temperatures, K, either a vector shared by all pressure levels or a
  matrix `temperature_K[i_T, i_p]` with one increasing column per pressure level, as in the
  ABSCO files;
- `axes.pressure_hPa`: increasing pressures, hPa.

`cross_section` interpolates linearly in each axis: in H2O, then in temperature on the grids
of the two pressure levels that bracket the pressure, then in pressure, then in wavenumber.
The table is therefore exact for any function that is linear in each axis. The H2O dry mole
fraction `q` of the conditions is converted to the wet mole fraction `q / (1 + q)` of the
table axis. Pressure, temperature and H2O outside the table are an error (no extrapolation);
an axis of length one is held constant. Depends on pressure, temperature and H2O.

```julia
axes = (wavenumber_cm = 6200.0:0.01:6210.0, h2o_mole_fraction = [0.0, 0.03],
        temperature_K = 200.0:10.0:320.0, pressure_hPa = [100.0, 500.0, 1000.0])
table = ABSCOTable(zeros(1001, 2, 13, 3), axes)
cross_section(table, 6201.0:0.002:6209.0, GasConditions(850.0, 290.0, 0.01))
```
Usually read with `read_absco_table(path)`.
"""
struct ABSCOTable{A<:AbstractArray{Float64,4}} <: AbstractCrossSectionModel
    sigma::A
    wavenumber_cm::Vector{Float64}
    h2o_mole_fraction::Vector{Float64}
    temperature_K::Matrix{Float64}
    pressure_hPa::Vector{Float64}
end

function ABSCOTable(sigma::AbstractArray{<:Real,4}, table_axes::NamedTuple)
    nu = collect(Float64, table_axes.wavenumber_cm)
    q = collect(Float64, table_axes.h2o_mole_fraction)
    p = collect(Float64, table_axes.pressure_hPa)
    T = temperature_matrix(table_axes.temperature_K, length(p))
    expected = (length(nu), length(q), size(T, 1), length(p))
    size(sigma) == expected || throw(DimensionMismatch(
        "ABSCOTable sigma has size $(size(sigma)); the axes give (nu, h2o, T, p) = $expected"))
    check_increasing("wavenumber_cm", nu)
    check_increasing("h2o_mole_fraction", q)
    check_increasing("pressure_hPa", p)
    for ip in 1:size(T, 2)
        check_increasing("temperature_K at pressure level $ip", view(T, :, ip))
    end
    length(nu) >= 2 || throw(ArgumentError("ABSCOTable needs at least two wavenumbers"))
    sigma64 = eltype(sigma) === Float64 ? sigma : Float64.(sigma)
    return ABSCOTable(sigma64, nu, q, T, p)
end

temperature_matrix(T::AbstractVector, n_p::Int) = repeat(collect(Float64, T), 1, n_p)
function temperature_matrix(T::AbstractMatrix, n_p::Int)
    size(T, 2) == n_p || throw(DimensionMismatch(
        "temperature_K matrix has $(size(T, 2)) columns; expected one per pressure level ($n_p)"))
    return Matrix{Float64}(T)
end

function check_increasing(name, v)
    all(i -> v[i] < v[i+1], 1:length(v)-1) ||
        throw(ArgumentError("ABSCOTable axis $name must be strictly increasing"))
    return nothing
end

depends_on(::ABSCOTable) = (:pressure, :temperature, :h2o)
spectral_range_cm(m::ABSCOTable) = (first(m.wavenumber_cm), last(m.wavenumber_cm))

"""
    bracket(grid, x, name, level = 0) -> (i, w)

Index `i` and weight `w` such that `x = (1 - w) * grid[i] + w * grid[i+1]` on an increasing
`grid`; for a grid of length one, `(1, 0)`. `x` outside the grid is a `DomainError`. The
weight keeps the type of `x`, so dual numbers carry their derivatives. `name` and `level` (a
pressure level, 0 for none) only label the error message.
"""
function bracket(grid::AbstractVector{Float64}, x::Real, name::AbstractString, level::Int = 0)
    n = length(grid)
    n == 1 && return (1, zero(x))
    xv = strip_duals(x)
    (grid[1] <= xv <= grid[n]) ||
        throw(DomainError(xv, string(name, " = ", xv, level > 0 ? " at pressure level $level" : "",
                                     " is outside the table range ($(grid[1]), $(grid[n]))")))
    i = clamp(searchsortedlast(grid, xv), 1, n - 1)
    w = (x - grid[i]) / (grid[i+1] - grid[i])
    return (i, w)
end

function cross_section(m::ABSCOTable, wavenumber_cm::AbstractVector, c::GasConditions)
    R = condition_eltype(c)
    p = R(c.pressure_hPa)
    T = R(c.temperature_K)
    q_dry = R(c.h2o_dry_mole_fraction)
    q_wet = q_dry / (1 + q_dry)
    check_grid_in_range(wavenumber_cm, spectral_range_cm(m))

    iq, wq = bracket(m.h2o_mole_fraction, q_wet, "h2o wet mole fraction")
    ip, wp = bracket(m.pressure_hPa, p, "pressure_hPa")
    # corners: (h2o index, temperature index, pressure index, weight)
    corners = absco_corners(m, iq, wq, ip, wp, T)

    sigma = zeros(R, length(wavenumber_cm))
    nu_grid = m.wavenumber_cm
    n_nu = length(nu_grid)
    @inbounds for (i, nu) in enumerate(wavenumber_cm)
        k = clamp(searchsortedlast(nu_grid, nu), 1, n_nu - 1)
        a = (nu - nu_grid[k]) / (nu_grid[k+1] - nu_grid[k])
        acc = zero(R)
        for (cq, cT, cp, w) in corners
            s = (1 - a) * m.sigma[k, cq, cT, cp] + a * m.sigma[k+1, cq, cT, cp]
            acc += w * s
        end
        sigma[i] = acc
    end
    return sigma
end

# The eight interpolation corners as a tuple, so the inner loop is unrolled and type stable.
function absco_corners(m::ABSCOTable, iq, wq::R, ip, wp::R, T::R) where {R}
    nq = length(m.h2o_mole_fraction)
    n_p = length(m.pressure_hPa)
    q2 = min(iq + 1, nq)
    p2 = min(ip + 1, n_p)
    iT1, wT1 = bracket(view(m.temperature_K, :, ip), T, "temperature_K", ip)
    iT2, wT2 = bracket(view(m.temperature_K, :, p2), T, "temperature_K", p2)
    nT = size(m.temperature_K, 1)
    t12, t22 = min(iT1 + 1, nT), min(iT2 + 1, nT)
    one_R = one(R)
    return ((iq, iT1, ip, (one_R - wq) * (one_R - wT1) * (one_R - wp)),
            (iq, t12, ip, (one_R - wq) * wT1 * (one_R - wp)),
            (q2, iT1, ip, wq * (one_R - wT1) * (one_R - wp)),
            (q2, t12, ip, wq * wT1 * (one_R - wp)),
            (iq, iT2, p2, (one_R - wq) * (one_R - wT2) * wp),
            (iq, t22, p2, (one_R - wq) * wT2 * wp),
            (q2, iT2, p2, wq * (one_R - wT2) * wp),
            (q2, t22, p2, wq * wT2 * wp))
end

export H2OBroadened, ABSCOTable
