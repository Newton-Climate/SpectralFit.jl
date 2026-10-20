# Interpolation from the fine wavenumber grid, shared by instruments.

"""
    interpolate_fine_grid(wavenumber_fine_cm::AbstractRange, values, query_cm)

Cubic B-spline (natural "Line" boundary, on grid) interpolation of `values` given on the
uniform fine grid, evaluated at `query_cm`, which may be unsorted and may contain dual
numbers. Same scheme as v0.1 `apply_instrument` (open question 4: kept for regression parity).
"""
function interpolate_fine_grid(wavenumber_fine_cm::AbstractRange, values::AbstractVector,
                               query_cm::AbstractVector)
    length(values) == length(wavenumber_fine_cm) ||
        throw(DimensionMismatch("values has length $(length(values)), fine grid has $(length(wavenumber_fine_cm))"))
    itp = Interpolations.interpolate(values,
              Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))))
    sitp = Interpolations.scale(itp, wavenumber_fine_cm)
    return map(q -> sitp(q), query_cm)
end

export interpolate_fine_grid
