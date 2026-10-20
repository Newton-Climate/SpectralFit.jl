# Noise estimation (section 3.9). Replaces v0.1 `calc_DCS_noise` and its hard-coded flat
# window with a per-record residual estimator (ported from
# DCS_experiments/DCS_retrievals/data_driven_noise.jl) and keeps the old shot-noise scaling
# model as `shot_noise_variance`.
#
# Ordering: the flat windows usually lie outside (or at the edges of) the retrieval window,
# so a noise model must be applied to the full-spectrum measurement, before `select_window`
# and before `time_average` (which then scales the variance by 1/count correctly):
#
#     ms = with_noise_variance(read_dcs_measurements(path, config), ResidualNoise(...))
#     m  = select_window(time_average(ms, Minute(15))[1], 6076.05, 6077.95)

# ---------------------------------------------------------------- polynomial residuals

"""
    _polynomial_residuals(x, y, order)

Residuals of the least-squares polynomial of degree `order` fitted to `(x, y)`. The
coordinate is mapped to [-1, 1] before building the Vandermonde matrix, so the fit is well
conditioned on wavenumbers near 6000 cm-1. Solved by QR (`\\`).
"""
function _polynomial_residuals(x::AbstractVector, y::AbstractVector, order::Integer)
    lo, hi = extrema(x)
    half = hi > lo ? (hi - lo) / 2 : one(float(hi))
    t = (x .- (lo + hi) / 2) ./ half
    A = [ti^k for ti in t, k in 0:order]
    coefficients = A \ collect(float.(y))
    return y .- A * coefficients
end

function _window_indices(coordinate, window)
    a, b = window
    return findall(c -> a < c < b, coordinate)       # open interval, as select_window
end

# ---------------------------------------------------------------- residual estimator

"""
    estimate_noise_variance(coordinate, intensity; flat_windows_cm, polynomial_order = 3)

Per-record noise variance (intensity units squared) from the residual of a low-order
polynomial fitted by least squares inside each spectrally flat window.

- `coordinate`: wavenumber in cm-1 (any sorted or unsorted vector).
- `intensity`: one record (vector, same length as `coordinate`), or a matrix with one record
  per row (`n_records x length(coordinate)`).
- `flat_windows_cm`: vector of `(minimum_cm, maximum_cm)` tuples, open intervals, free of
  absorption lines.
- `polynomial_order`: degree of the detrending polynomial in each window.

A separate polynomial is fitted in each window (a window with fewer than
`polynomial_order + 2` points is skipped) and the residuals are pooled:
`sum(r.^2) / (n_points - n_coefficients)`, an unbiased white-noise variance. Returns a
scalar for a vector `intensity` and a `Vector` (one value per row) for a matrix.

Deviation from data_driven_noise.jl: that script fitted one polynomial across all windows
together and used `var(r)`; per-window fits do not assume the baseline is one polynomial
across the gap, and the degree-of-freedom correction removes the small low bias.

The result is a single variance for all observations of the record (white noise). Apply it
to the full-spectrum measurement before `select_window` (see the file header).

# Example
```julia
nu = collect(6076.0:0.0067:6078.0)
sigma2 = estimate_noise_variance(nu, intensity;
                                 flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)],
                                 polynomial_order = 3)
```
"""
function estimate_noise_variance(coordinate::AbstractVector, intensity::AbstractVector;
                                 flat_windows_cm, polynomial_order::Integer = 3)
    length(coordinate) == length(intensity) ||
        throw(DimensionMismatch("coordinate has length $(length(coordinate)), intensity $(length(intensity))"))
    polynomial_order >= 0 || throw(ArgumentError("polynomial_order must be non-negative"))
    sum_squares = 0.0
    dof = 0
    for window in flat_windows_cm
        idx = _window_indices(coordinate, window)
        length(idx) >= polynomial_order + 2 || continue
        r = _polynomial_residuals(coordinate[idx], intensity[idx], polynomial_order)
        sum_squares += sum(abs2, r)
        dof += length(idx) - (polynomial_order + 1)
    end
    dof > 0 || throw(ArgumentError("estimate_noise_variance: no flat window holds at least " *
                                   "$(polynomial_order + 2) points of the coordinate"))
    return sum_squares / dof
end

function estimate_noise_variance(coordinate::AbstractVector, intensity::AbstractMatrix;
                                 flat_windows_cm, polynomial_order::Integer = 3)
    size(intensity, 2) == length(coordinate) ||
        throw(DimensionMismatch("intensity must be n_records x length(coordinate)"))
    return [estimate_noise_variance(coordinate, view(intensity, i, :);
                                    flat_windows_cm = flat_windows_cm,
                                    polynomial_order = polynomial_order)
            for i in axes(intensity, 1)]
end

# ---------------------------------------------------------------- shot-noise model (v0.1)

"""
    shot_noise_variance(coordinate, intensity::AbstractMatrix;
                        flat_window_cm = (6255.1, 6255.4), polynomial_order = 3) -> Vector

The v0.1 `calc_DCS_noise` model: noise standard deviation proportional to the square root of
the mean signal of each record. Steps, as in v0.1:

1. In `flat_window_cm` fit a polynomial of degree `polynomial_order` to each record and take
   the residual.
2. Remove the mean residual over records at each point (fixed-pattern structure), then take
   the standard deviation of each record's residual, `s_i`.
3. Fit `s_i = a * sqrt(S_i)` by least squares through the origin, where `S_i` is the mean of
   record `i` over the whole coordinate.
4. Return `(a * sqrt(S_i))^2` for each record, in intensity units squared.

`intensity` has one record per row (`n_records x length(coordinate)`); at least two records
are needed because step 2 removes the across-record mean (which biases the variance low by
about `(n_records - 1) / n_records`, kept for v0.1 parity). The default window is the v0.1
hard-coded one, which over-estimates the noise by about 1.5 times on 20160921 according to
data_driven_noise.jl; prefer `estimate_noise_variance` with windows near the retrieval window.

# Example
```julia
sigma2 = shot_noise_variance(nu, intensity)            # v0.1 behaviour
```
"""
function shot_noise_variance(coordinate::AbstractVector, intensity::AbstractMatrix;
                             flat_window_cm = (6255.1, 6255.4), polynomial_order::Integer = 3)
    size(intensity, 2) == length(coordinate) ||
        throw(DimensionMismatch("intensity must be n_records x length(coordinate)"))
    n_records = size(intensity, 1)
    n_records >= 2 || throw(ArgumentError("shot_noise_variance needs at least two records"))
    idx = _window_indices(coordinate, flat_window_cm)
    length(idx) >= polynomial_order + 2 ||
        throw(ArgumentError("shot_noise_variance: flat window $flat_window_cm holds " *
                            "$(length(idx)) points, need at least $(polynomial_order + 2)"))
    x = coordinate[idx]
    residuals = reduce(vcat, (_polynomial_residuals(x, intensity[i, idx], polynomial_order)'
                              for i in 1:n_records))
    residuals = residuals .- mean(residuals; dims = 1)
    s = vec(std(residuals; dims = 2))
    root_signal = sqrt.(vec(mean(intensity; dims = 2)))
    slope = dot(root_signal, s) / dot(root_signal, root_signal)
    return (slope .* root_signal) .^ 2
end

# ---------------------------------------------------------------- noise models

"""
Supertype of noise models. A subtype implements
`record_noise_variances(model, coordinate, intensity::AbstractMatrix) -> Vector` (one
variance per row of `intensity`, in intensity units squared).
"""
abstract type AbstractNoiseModel end

"""
    ResidualNoise(; flat_windows_cm, polynomial_order = 3)

Noise model using `estimate_noise_variance` on each record independently.

# Example
```julia
ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)], polynomial_order = 3)
```
"""
struct ResidualNoise <: AbstractNoiseModel
    flat_windows_cm::Vector{Tuple{Float64,Float64}}
    polynomial_order::Int
end
function ResidualNoise(; flat_windows_cm, polynomial_order::Integer = 3)
    windows = [(Float64(a), Float64(b)) for (a, b) in flat_windows_cm]
    isempty(windows) && throw(ArgumentError("ResidualNoise needs at least one flat window"))
    all(w -> w[1] < w[2], windows) || throw(ArgumentError("each flat window needs minimum < maximum"))
    return ResidualNoise(windows, polynomial_order)
end

"""
    ShotNoise(; flat_window_cm = (6255.1, 6255.4), polynomial_order = 3)

Noise model using `shot_noise_variance` (the v0.1 `calc_DCS_noise` model). It needs the whole
series of records, so apply it to a vector of measurements, not to one measurement.

# Example
```julia
ms = with_noise_variance(ms, ShotNoise())
```
"""
struct ShotNoise <: AbstractNoiseModel
    flat_window_cm::Tuple{Float64,Float64}
    polynomial_order::Int
end
ShotNoise(; flat_window_cm = (6255.1, 6255.4), polynomial_order::Integer = 3) =
    ShotNoise((Float64(flat_window_cm[1]), Float64(flat_window_cm[2])), polynomial_order)

"""
    ConstantNoise(variance)

Noise model giving every observation the same `variance` (intensity units squared, > 0).

# Example
```julia
ConstantNoise(1e-6)
```
"""
struct ConstantNoise <: AbstractNoiseModel
    variance::Float64
    function ConstantNoise(variance::Real)
        variance > 0 || throw(ArgumentError("ConstantNoise variance must be positive"))
        return new(Float64(variance))
    end
end

"""
    record_noise_variances(model::AbstractNoiseModel, coordinate, intensity::AbstractMatrix) -> Vector

One noise variance per record (row of `intensity`), in intensity units squared.

# Example
```julia
record_noise_variances(ConstantNoise(1e-6), nu, intensity)   # fill(1e-6, size(intensity, 1))
```
"""
record_noise_variances(model::ResidualNoise, coordinate, intensity::AbstractMatrix) =
    estimate_noise_variance(coordinate, intensity; flat_windows_cm = model.flat_windows_cm,
                            polynomial_order = model.polynomial_order)
record_noise_variances(model::ShotNoise, coordinate, intensity::AbstractMatrix) =
    shot_noise_variance(coordinate, intensity; flat_window_cm = model.flat_window_cm,
                        polynomial_order = model.polynomial_order)
record_noise_variances(model::ConstantNoise, coordinate, intensity::AbstractMatrix) =
    fill(model.variance, size(intensity, 1))

"""
    with_noise_variance(measurement::Measurement, model::AbstractNoiseModel) -> Measurement
    with_noise_variance(measurements::AbstractVector{<:Measurement}, model) -> Vector{Measurement}

New measurement(s) identical to the input except for `noise_variance`, which is replaced by
the model's per-record variance expanded to every observation. Apply to full-spectrum
measurements before `select_window` and `time_average`. `ShotNoise` needs the vector form
(the scaling is fitted across records); the vector form requires a shared coordinate axis.

# Example
```julia
m2 = with_noise_variance(m, ResidualNoise(flat_windows_cm = [(6076.10, 6076.60)]))
```
"""
function with_noise_variance(m::Measurement, model::AbstractNoiseModel)
    model isa ShotNoise &&
        throw(ArgumentError("ShotNoise is fitted across records: pass a vector of measurements"))
    return only(with_noise_variance([m], model))
end

function with_noise_variance(measurements::AbstractVector{<:Measurement}, model::AbstractNoiseModel)
    isempty(measurements) && return measurements
    coordinate = first(measurements).coordinate
    all(m -> m.coordinate == coordinate, measurements) ||
        throw(ArgumentError("with_noise_variance: measurements do not share one coordinate axis"))
    intensity = reduce(vcat, (m.y' for m in measurements))
    variances = record_noise_variances(model, coordinate, intensity)
    all(v -> v > 0 && isfinite(v), variances) ||
        throw(ArgumentError("with_noise_variance: estimated noise variance is not positive " *
                            "and finite (noise-free data or bad flat windows?)"))
    return [Measurement(y = m.y, coordinate = m.coordinate, noise_variance = v, time = m.time,
                        conditions = m.conditions, metadata = m.metadata)
            for (m, v) in zip(measurements, variances)]
end

export estimate_noise_variance, shot_noise_variance
export AbstractNoiseModel, ResidualNoise, ShotNoise, ConstantNoise
export record_noise_variances, with_noise_variance
