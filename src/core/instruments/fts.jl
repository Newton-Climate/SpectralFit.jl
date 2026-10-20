# Fourier transform spectrometer (section 3.6 of docs/architecture_plan.md). The instrument
# line shape (ILS) is applied as a sparse matrix built once by `bind_fine_grid`; the matrix
# helper `line_shape_matrix` is shared with grating.jl.
#
# Convention: the modelled spectrum at observed wavenumber nu_i is
#     y_i = int T(nu) K_i(nu - nu_i) dnu,
# with the kernel K_i sampled on the fine grid (trapezoid rule) and each row normalized to sum
# to 1, which compensates the truncation of the slowly decaying sinc.

"""Supertype of FTS apodization functions."""
abstract type AbstractApodization end

"""
    Boxcar()

No apodization: the ILS of maximum optical path difference `L` (cm) is
`2L sinc(2L delta)`, with `sinc(u) = sin(pi u) / (pi u)`; full width at half maximum
`1.2067 / (2L)` cm-1.
"""
struct Boxcar <: AbstractApodization end

"""
    Triangular()

Triangular apodization: ILS `L sinc(L delta)^2`, full width at half maximum `1.7718 / (2L)`.
"""
struct Triangular <: AbstractApodization end

unshifted_ils(::Boxcar, L, delta) = 2L * sinc(2L * delta)
unshifted_ils(::Triangular, L, delta) = L * sinc(L * delta)^2

"""
    FTS(; max_path_difference_cm, apodization = Boxcar(), field_of_view_rad = 0.0,
        ils_half_width_cm = 50.5 / (2 * max_path_difference_cm))

Fourier transform spectrometer. `max_path_difference_cm` is the maximum optical path
difference L (cm), `field_of_view_rad` the half-angle theta (rad) of a circular field of view,
which convolves the ILS with a box of width `nu theta^2 / 2` towards higher true wavenumber,
so lines appear shifted down by `nu theta^2 / 4`. The ILS is truncated at
`ils_half_width_cm` (cm-1; the default ends 50.5 sinc lobes out, between two zeros).

```julia
fts = FTS(max_path_difference_cm = 45.0)
p = bind_fine_grid(prepare(fts, measurement), default_fine_grid(prepare(fts, measurement)))
y = apply_instrument(p, grid, transmission, x)
```
"""
struct FTS{A<:AbstractApodization} <: AbstractInstrument
    max_path_difference_cm::Float64
    apodization::A
    field_of_view_rad::Float64
    ils_half_width_cm::Float64
end
function FTS(; max_path_difference_cm::Real, apodization::AbstractApodization = Boxcar(),
             field_of_view_rad::Real = 0.0,
             ils_half_width_cm::Real = 50.5 / (2 * max_path_difference_cm))
    max_path_difference_cm > 0 || throw(ArgumentError("FTS: max_path_difference_cm must be positive"))
    field_of_view_rad >= 0 || throw(ArgumentError("FTS: field_of_view_rad must be non-negative"))
    ils_half_width_cm > 0 || throw(ArgumentError("FTS: ils_half_width_cm must be positive"))
    return FTS(Float64(max_path_difference_cm), apodization, Float64(field_of_view_rad),
               Float64(ils_half_width_cm))
end

"""
    instrument_line_shape(fts::FTS, delta_cm, wavenumber_cm = 0.0)

ILS value (1/cm-1) at offset `delta_cm = nu_true - nu_observed`. With a field of view the
ILS depends on `wavenumber_cm` and is the average of the apodized sinc over the box
`[0, wavenumber_cm theta^2 / 2]` (16-point midpoint rule). Integrates to 1 over all offsets.

```julia
fts = FTS(max_path_difference_cm = 1.0)
instrument_line_shape(fts, 0.0)   # 2.0
```
"""
function instrument_line_shape(fts::FTS, delta_cm::Real, wavenumber_cm::Real = 0.0)
    L = fts.max_path_difference_cm
    width = fov_box_width_cm(fts, wavenumber_cm)
    width == 0 && return unshifted_ils(fts.apodization, L, delta_cm)
    n = 16
    return sum(unshifted_ils(fts.apodization, L, delta_cm - (k - 0.5) * width / n) for k in 1:n) / n
end

fov_box_width_cm(fts::FTS, wavenumber_cm) = wavenumber_cm * fts.field_of_view_rad^2 / 2

"""
    PreparedFTS

`prepare(::FTS, measurement)` result: the instrument, the observed wavenumbers (cm-1), and
after `bind_fine_grid` the fine grid and the sparse ILS matrix (`nothing` before).
"""
struct PreparedFTS{A, G, M}
    instrument::FTS{A}
    wavenumber_cm::Vector{Float64}
    wavenumber_fine_cm::G
    matrix::M
end

"""
    prepare(fts::FTS, measurement) -> PreparedFTS

Store the observed wavenumbers (`measurement.coordinate`, cm-1). The ILS matrix is built by
`bind_fine_grid` once the fine grid is known.
"""
prepare(fts::FTS, m::Measurement) = PreparedFTS(fts, collect(Float64, m.coordinate), nothing, nothing)

n_observations(p::PreparedFTS) = length(p.wavenumber_cm)

function wavenumber_coverage_cm(p::PreparedFTS, x_prior = nothing)
    fts = p.instrument
    hw = fts.ils_half_width_cm
    hi = maximum(p.wavenumber_cm)
    return (minimum(p.wavenumber_cm) - hw, hi + hw + fov_box_width_cm(fts, hi))
end

"""
    default_fine_grid(prepared::PreparedFTS; oversampling = 10)

Uniform grid over `wavenumber_coverage_cm` with step `1 / (2L oversampling)` cm-1, which
samples the ILS. Gas lines narrower than the ILS may need a finer grid.
"""
function default_fine_grid(p::PreparedFTS; oversampling::Real = 10)
    lo, hi = wavenumber_coverage_cm(p)
    h = 1 / (2 * p.instrument.max_path_difference_cm * oversampling)
    return range(lo; step = h, length = ceil(Int, (hi - lo) / h) + 1)
end

"""
    line_shape_matrix(observed_cm, grid, lo_offset_cm, hi_offset_cm, kernel) -> SparseMatrixCSC

Rows: observed wavenumbers; columns: points of the uniform fine `grid`. Row i holds
`kernel(i, grid[j] - observed_cm[i])` for offsets in `[lo_offset_cm, hi_offset_cm]` (midpoint
rule; a point whose cell straddles an end of the support is weighted by the part inside),
normalized to sum 1. The kernel may be evaluated up to half a grid step outside the support. Throws if the grid does not cover every row's support.
"""
function line_shape_matrix(observed_cm::AbstractVector, grid::AbstractRange, lo_offset_cm::Real,
                           hi_offset_cm::Real, kernel)
    h = step(grid)
    N = length(grid)
    tol = 1e-9 * h
    rows = Int[]; cols = Int[]; vals = Float64[]
    for (i, nu) in enumerate(observed_cm)
        (nu + lo_offset_cm >= first(grid) - h / 2 && nu + hi_offset_cm <= last(grid) + h / 2) ||
            throw(ArgumentError("line_shape_matrix: fine grid ($(first(grid)), $(last(grid))) does not cover $(nu + lo_offset_cm) to $(nu + hi_offset_cm) cm-1"))
        # midpoint rule: each fine point carries the part of its cell inside the support
        jlo = max(1, ceil(Int, (nu + lo_offset_cm - h / 2 - first(grid) + tol) / h) + 1)
        jhi = min(N, floor(Int, (nu + hi_offset_cm + h / 2 - first(grid) - tol) / h) + 1)
        w = map(jlo:jhi) do j
            d = grid[j] - nu
            kernel(i, d) * (min(d + h / 2, hi_offset_cm) - max(d - h / 2, lo_offset_cm)) / h
        end
        s = sum(w)
        s > 0 || throw(ArgumentError("line_shape_matrix: line shape sums to $s for observation $i; is the fine grid too coarse?"))
        append!(rows, fill(i, length(w)))
        append!(cols, jlo:jhi)
        append!(vals, w ./ s)
    end
    return sparse(rows, cols, vals, length(observed_cm), N)
end

"""
    bind_fine_grid(prepared::PreparedFTS, wavenumber_fine_cm::AbstractRange) -> PreparedFTS

Build the sparse ILS matrix from the fine grid to the observed wavenumbers.
"""
function bind_fine_grid(p::PreparedFTS, grid::AbstractRange)
    fts = p.instrument
    hw = fts.ils_half_width_cm
    nu = p.wavenumber_cm
    kernel = (i, delta) -> instrument_line_shape(fts, delta, nu[i])
    A = line_shape_matrix(nu, grid, -hw, hw + fov_box_width_cm(fts, maximum(nu)), kernel)
    return PreparedFTS(fts, nu, grid, A)
end

"""
    apply_instrument(prepared::PreparedFTS, wavenumber_fine_cm, transmission, x)

ILS-convolved transmission at the observed wavenumbers: the stored sparse matrix times
`transmission`. Without a bound grid the matrix is built on the fly (slow; the builder binds
it once).
"""
function apply_instrument(p::PreparedFTS{<:Any,<:AbstractRange,<:SparseMatrixCSC},
                          grid::AbstractRange, transmission::AbstractVector, x)
    check_bound_grid(p.wavenumber_fine_cm, grid, transmission)
    return p.matrix * transmission
end
apply_instrument(p::PreparedFTS{<:Any,Nothing,Nothing}, grid::AbstractRange,
                 transmission::AbstractVector, x) =
    apply_instrument(bind_fine_grid(p, grid), grid, transmission, x)

function check_bound_grid(bound::AbstractRange, grid::AbstractRange, transmission)
    length(transmission) == length(grid) ||
        throw(DimensionMismatch("transmission has length $(length(transmission)), fine grid has $(length(grid))"))
    (length(grid) == length(bound) && isapprox(first(grid), first(bound); atol = 1e-9 * step(bound)) &&
     isapprox(step(grid), step(bound); rtol = 1e-9)) ||
        throw(ArgumentError("fine grid differs from the one passed to bind_fine_grid"))
    return nothing
end

export AbstractApodization, Boxcar, Triangular, FTS, PreparedFTS, instrument_line_shape
export line_shape_matrix
