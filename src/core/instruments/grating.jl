# Grating spectrometer with a tabulated instrument line shape (section 3.6 of
# docs/architecture_plan.md). Same sparse-matrix form as the FTS (`line_shape_matrix` in
# fts.jl): y_i = sum_j A[i, j] T(nu_j), each row normalized to sum 1.

"""
    GratingSpectrometer(; offsets_cm, response)

Grating spectrometer whose line shape is tabulated at `offsets_cm` (sorted, cm-1, offset of
the fine wavenumber from the pixel centre, `nu_fine - nu_pixel`). `response` is a vector (one
line shape for every pixel) or a matrix with one column per pixel. Between table points the
line shape is a cubic B-spline when `offsets_cm` is an `AbstractRange`, linear otherwise; it
is zero outside the table. Normalization is irrelevant: each pixel's weights are scaled to
sum 1.

```julia
offsets = range(-0.3, 0.3; length = 601)
grating = GratingSpectrometer(offsets_cm = offsets, response = exp.(-offsets .^ 2 ./ (2 * 0.05^2)))
```
"""
struct GratingSpectrometer{O<:AbstractVector{Float64}, R<:AbstractVecOrMat{Float64}} <: AbstractInstrument
    offsets_cm::O
    response::R
    function GratingSpectrometer(offsets_cm::AbstractVector, response::AbstractVecOrMat)
        o = offsets_cm isa AbstractRange ? range(Float64(first(offsets_cm)), Float64(last(offsets_cm));
                                                 length = length(offsets_cm)) :
            collect(Float64, offsets_cm)
        length(o) >= 2 || throw(ArgumentError("GratingSpectrometer: need at least two offsets"))
        issorted(o) && allunique(o) || throw(ArgumentError("GratingSpectrometer: offsets_cm must be strictly increasing"))
        size(response, 1) == length(o) ||
            throw(DimensionMismatch("GratingSpectrometer: response has $(size(response, 1)) rows, offsets_cm has $(length(o))"))
        r = Array{Float64}(response)
        all(isfinite, r) || throw(ArgumentError("GratingSpectrometer: response must be finite"))
        return new{typeof(o),typeof(r)}(o, r)
    end
end
GratingSpectrometer(; offsets_cm, response) = GratingSpectrometer(offsets_cm, response)

table_interpolant(offsets::AbstractRange, values::AbstractVector) =
    Interpolations.scale(Interpolations.interpolate(collect(values),
        Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid())))),
        offsets)
table_interpolant(offsets::AbstractVector, values::AbstractVector) =
    Interpolations.interpolate((collect(offsets),), collect(values), Interpolations.Gridded(Interpolations.Linear()))

"""
    PreparedGrating

`prepare(::GratingSpectrometer, measurement)` result: the instrument, pixel wavenumbers
(cm-1), and after `bind_fine_grid` the fine grid and sparse line-shape matrix.
"""
struct PreparedGrating{I<:GratingSpectrometer, G, M}
    instrument::I
    wavenumber_cm::Vector{Float64}
    wavenumber_fine_cm::G
    matrix::M
end

"""
    prepare(grating::GratingSpectrometer, measurement) -> PreparedGrating

Store the pixel wavenumbers (`measurement.coordinate`, cm-1); checks one response column per
pixel when `response` is a matrix.
"""
function prepare(g::GratingSpectrometer, m::Measurement)
    nu = collect(Float64, m.coordinate)
    g.response isa AbstractMatrix && size(g.response, 2) != length(nu) &&
        throw(DimensionMismatch("GratingSpectrometer: response has $(size(g.response, 2)) columns, measurement has $(length(nu)) pixels"))
    return PreparedGrating(g, nu, nothing, nothing)
end

n_observations(p::PreparedGrating) = length(p.wavenumber_cm)

wavenumber_coverage_cm(p::PreparedGrating, x_prior = nothing) =
    (minimum(p.wavenumber_cm) + first(p.instrument.offsets_cm),
     maximum(p.wavenumber_cm) + last(p.instrument.offsets_cm))

"""
    default_fine_grid(prepared::PreparedGrating; oversampling = 1)

Uniform grid over `wavenumber_coverage_cm` with the smallest table spacing divided by
`oversampling` as step (cm-1).
"""
function default_fine_grid(p::PreparedGrating; oversampling::Real = 1)
    lo, hi = wavenumber_coverage_cm(p)
    h = minimum(diff(p.instrument.offsets_cm)) / oversampling
    return range(lo; step = h, length = ceil(Int, (hi - lo) / h) + 1)
end

"""
    bind_fine_grid(prepared::PreparedGrating, wavenumber_fine_cm::AbstractRange) -> PreparedGrating

Build the sparse line-shape matrix from the fine grid to the pixel wavenumbers.
"""
function bind_fine_grid(p::PreparedGrating, grid::AbstractRange)
    g = p.instrument
    lo, hi = first(g.offsets_cm), last(g.offsets_cm)
    clampd(d) = clamp(d, lo, hi)
    A = if g.response isa AbstractVector
        itp = table_interpolant(g.offsets_cm, g.response)
        line_shape_matrix(p.wavenumber_cm, grid, lo, hi, (i, d) -> itp(clampd(d)))
    else
        itps = [table_interpolant(g.offsets_cm, view(g.response, :, i)) for i in axes(g.response, 2)]
        line_shape_matrix(p.wavenumber_cm, grid, lo, hi, (i, d) -> itps[i](clampd(d)))
    end
    return PreparedGrating(g, p.wavenumber_cm, grid, A)
end

"""
    apply_instrument(prepared::PreparedGrating, wavenumber_fine_cm, transmission, x)

Line-shape-convolved transmission at the pixel wavenumbers (sparse matrix times
`transmission`). Without a bound grid the matrix is built on the fly.
"""
function apply_instrument(p::PreparedGrating{<:Any,<:AbstractRange,<:SparseMatrixCSC},
                          grid::AbstractRange, transmission::AbstractVector, x)
    check_bound_grid(p.wavenumber_fine_cm, grid, transmission)
    return p.matrix * transmission
end
apply_instrument(p::PreparedGrating{<:Any,Nothing,Nothing}, grid::AbstractRange,
                 transmission::AbstractVector, x) =
    apply_instrument(bind_fine_grid(p, grid), grid, transmission, x)

export GratingSpectrometer, PreparedGrating
