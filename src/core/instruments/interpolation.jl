# Interpolation from the fine wavenumber grid, shared by instruments: the cubic B-spline
# evaluated per call, and the same map stored as a sparse matrix for fixed query points.

const FINE_GRID_SPLINE = Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())))

"""
    interpolate_fine_grid(wavenumber_fine_cm::AbstractRange, values, query_cm)

Cubic B-spline (natural "Line" boundary, on grid) interpolation of `values` given on the
uniform fine grid, evaluated at `query_cm`, which may be unsorted and may contain dual
numbers. Same scheme as v0.1 `apply_instrument` (open question 4: kept for regression
parity). For query points that do not depend on the state, `interpolation_matrix` gives the
same map as a sparse matrix.
"""
function interpolate_fine_grid(wavenumber_fine_cm::AbstractRange, values::AbstractVector,
                               query_cm::AbstractVector)
    length(values) == length(wavenumber_fine_cm) ||
        throw(DimensionMismatch("values has length $(length(values)), fine grid has $(length(wavenumber_fine_cm))"))
    itp = Interpolations.interpolate(values, FINE_GRID_SPLINE)
    sitp = Interpolations.scale(itp, wavenumber_fine_cm)
    return map(q -> sitp(q), query_cm)
end

"""
    interpolation_matrix(wavenumber_fine_cm::AbstractRange, query_cm; half_width = 40,
                         drop_tol = 1e-15) -> SparseMatrixCSC

Sparse matrix `W` (rows: `query_cm`, columns: fine-grid points) with
`W * values ≈ interpolate_fine_grid(wavenumber_fine_cm, values, query_cm)` to machine
precision. The interpolating B-spline needs a global prefilter solve, so the exact map is
dense, but its weights decay like `0.268^d` with the distance `d` (fine-grid points) between
query and column. Column `j` is therefore the spline of the unit vector `e_j` on the
sub-grid `j - half_width : j + half_width` (clipped at the ends of the grid, where the true
"Line" boundary is kept), evaluated at the queries inside that sub-grid. The truncation error
is about `0.268^half_width` (1e-23 for 40); entries below `drop_tol` are dropped. Interior
columns share one sub-grid spline, so the cost is O(`length(query_cm) * half_width`) plus
O(`half_width^2`) for the columns near the ends. Throws an `ArgumentError` for a query
outside the grid.

```julia
grid = range(6075.0, 6079.0; step = 0.002)
W = interpolation_matrix(grid, [6076.3, 6077.1])
W * values ≈ interpolate_fine_grid(grid, values, [6076.3, 6077.1])
```
"""
function interpolation_matrix(grid::AbstractRange, query_cm::AbstractVector{<:Real};
                              half_width::Integer = 40, drop_tol::Real = 1e-15)
    n = length(grid)
    K = Int(half_width)
    K >= 1 || throw(ArgumentError("interpolation_matrix: half_width must be positive, got $K"))
    unit_spline(len, k) = Interpolations.interpolate([i == k ? 1.0 : 0.0 for i in 1:len], FINE_GRID_SPLINE)
    interior = unit_spline(2K + 1, K + 1)
    near_edge = Dict{Int,typeof(interior)}()   # columns whose sub-grid is clipped by an end
    rows = Int[]; cols = Int[]; vals = Float64[]
    sizehint!(rows, length(query_cm) * K); sizehint!(cols, length(query_cm) * K)
    sizehint!(vals, length(query_cm) * K)
    for (i, q) in enumerate(query_cm)
        t = Interpolations.coordlookup(grid, q)   # fractional fine-grid index, as in `scale`
        (isfinite(t) && 1 - 1e-9 <= t <= n + 1e-9) ||
            throw(ArgumentError("interpolation_matrix: query $q cm-1 is outside the fine grid ($(first(grid)), $(last(grid)))"))
        t = clamp(t, 1.0, Float64(n))
        for j in max(1, ceil(Int, t - K)):min(n, floor(Int, t + K))
            lo = max(1, j - K); hi = min(n, j + K)
            itp = (lo == j - K && hi == j + K) ? interior :
                  get!(() -> unit_spline(hi - lo + 1, j - lo + 1), near_edge, j)
            w = itp(clamp(t - lo + 1, 1.0, Float64(hi - lo + 1)))
            if abs(w) >= drop_tol
                push!(rows, i); push!(cols, j); push!(vals, w)
            end
        end
    end
    return sparse(rows, cols, vals, length(query_cm), n)
end

export interpolate_fine_grid, interpolation_matrix
