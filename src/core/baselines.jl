# Measurement spaces and baselines (section 3.8 of docs/architecture_plan.md).

# ---------------------------------------------------------------- measurement space

"""
    to_measurement_space(space, instrument_output)

`LinearSpace()`: the instrument output unchanged. `LogSpace()`: its natural log.

```julia
to_measurement_space(LogSpace(), [1.0, exp(1)])   # [0.0, 1.0]
```
"""
to_measurement_space(::LinearSpace, y::AbstractVector) = y
to_measurement_space(::LogSpace, y::AbstractVector) = log.(y)

"""
    observation_vector(space, measurement)

The stored observation `measurement.y` (instrument units) in `space`, as a new vector. The
measurement itself is never modified.

```julia
m = Measurement(y = [2.0, 3.0], coordinate = [6076.0, 6076.1], noise_variance = 1e-4)
observation_vector(LogSpace(), m)   # log.([2.0, 3.0])
```
"""
observation_vector(::LinearSpace, m::Measurement) = copy(m.y)
observation_vector(::LogSpace, m::Measurement) = log.(m.y)

"""
    obs_precision(space, measurement; masked_windows_cm = Tuple{Float64,Float64}[],
                  mask_factor = 1e-5) -> Diagonal

Inverse observation covariance (diagonal). `LinearSpace`: `1 / noise_variance`. `LogSpace`:
`y^2 / noise_variance` by first-order propagation of the log, always from the stored linear
`y`. Observations inside any closed interval `[minimum_cm, maximum_cm]` of `masked_windows_cm`
(in units of `measurement.coordinate`) are de-weighted by `mask_factor` (open question 8,
the v0.1 behaviour).

```julia
m = Measurement(y = [2.0, 3.0], coordinate = [6076.0, 6076.1], noise_variance = 1e-4)
obs_precision(LogSpace(), m; masked_windows_cm = [(6076.05, 6076.2)])   # Diagonal([4e4, 9e-1])
```
"""
function obs_precision(space::AbstractMeasurementSpace, m::Measurement;
                       masked_windows_cm = Tuple{Float64,Float64}[], mask_factor::Real = 1e-5)
    mask_factor > 0 || throw(ArgumentError("mask_factor must be positive, got $mask_factor"))
    precision = linear_precision(space, m)
    for i in find_mask(m.coordinate, masked_windows_cm)
        precision[i] *= mask_factor
    end
    return Diagonal(precision)
end

linear_precision(::LinearSpace, m::Measurement) = 1 ./ m.noise_variance
function linear_precision(::LogSpace, m::Measurement)
    all(v -> v > 0, m.y) ||
        throw(ArgumentError("LogSpace needs a positive measurement; y has non-positive values"))
    return m.y .^ 2 ./ m.noise_variance
end

"""
    find_mask(coordinate, masked_windows) -> Vector{Int}

Indices of `coordinate` inside any closed interval of `masked_windows`, given either as a
collection of `(minimum, maximum)` pairs or as an n-by-2 matrix (v0.1 form). As in v0.1
`find_mask`, indices are concatenated per window, but edges are included (v0.1 excluded
them). Internal.
"""
function find_mask(coordinate::AbstractVector, masked_windows::AbstractMatrix)
    size(masked_windows, 2) == 2 ||
        throw(DimensionMismatch("masked_windows matrix must have two columns"))
    return find_mask(coordinate, [(masked_windows[j, 1], masked_windows[j, 2])
                                  for j in axes(masked_windows, 1)])
end
function find_mask(coordinate::AbstractVector, masked_windows)
    indices = Int[]
    for (lo, hi) in masked_windows
        append!(indices, findall(c -> lo <= c <= hi, coordinate))
    end
    return indices
end

# ---------------------------------------------------------------- Legendre polynomials

"""
    legendre_matrix(n_points, n_coefficients) -> Matrix{Float64}

Legendre polynomials `P_0 ... P_{n_coefficients-1}` evaluated on
`range(-1, 1, length = n_points)`, one column per polynomial (`n_points` by
`n_coefficients`). The transpose of v0.1 `compute_legendre_poly`, with the same recurrence,
and also valid for one coefficient.

```julia
P = legendre_matrix(5, 3)
P[:, 2] == collect(range(-1, 1, length = 5))   # true
```
"""
function legendre_matrix(n_points::Integer, n_coefficients::Integer)
    n_points >= 2 || throw(ArgumentError("legendre_matrix: need at least 2 points, got $n_points"))
    n_coefficients >= 1 ||
        throw(ArgumentError("legendre_matrix: need at least 1 coefficient, got $n_coefficients"))
    x = collect(range(-1, stop = 1, length = n_points))
    P = zeros(n_points, n_coefficients)
    P[:, 1] .= 1.0
    n_coefficients >= 2 && (P[:, 2] .= x)
    for n in 2:n_coefficients-1
        l = n - 1
        for i in eachindex(x)
            P[i, n+1] = ((2l + 1) * x[i] * P[i, n] - l * P[i, n-1]) / (l + 1)
        end
    end
    return P
end

# ---------------------------------------------------------------- baselines

"""
    LegendreBaseline(n_coefficients; mode = Multiplicative(), name = :window_1)
    LegendreBaseline{Mode, Name}(n_coefficients)

Polynomial baseline with `n_coefficients` Legendre terms (one is a constant) over the
observation axis of a window. `Mode` is `Multiplicative` or `Additive`; `Name` is the entry
under `x.baseline` holding the coefficients (`x.baseline.window_1`). `prepare(baseline,
measurement)` builds the Legendre matrix once.

```julia
b = prepare(LegendreBaseline(5; name = :ch4_window), measurement)
apply_baseline(b, LinearSpace(), y, x)   # y .* (P * x.baseline.ch4_window)
```
"""
struct LegendreBaseline{Mode, Name} <: AbstractBaseline
    n_coefficients::Int
    function LegendreBaseline{Mode,Name}(n::Integer) where {Mode,Name}
        Mode in (Multiplicative, Additive) ||
            throw(ArgumentError("LegendreBaseline mode must be Multiplicative or Additive, got $Mode"))
        Name isa Symbol || throw(ArgumentError("LegendreBaseline name must be a Symbol, got $Name"))
        n >= 1 || throw(ArgumentError("LegendreBaseline: need at least 1 coefficient, got $n"))
        return new{Mode,Name}(n)
    end
end
LegendreBaseline(n_coefficients::Integer; mode = Multiplicative(), name::Symbol = :window_1) =
    LegendreBaseline{typeof(mode),name}(n_coefficients)

"""
    PreparedLegendreBaseline{Mode, Name}

A `LegendreBaseline` with its Legendre matrix (`n_observations` by `n_coefficients`) built
for one measurement window by `prepare`.

```julia
b = prepare(LegendreBaseline(3), measurement)
size(b.matrix)   # (length(measurement.y), 3)
```
"""
struct PreparedLegendreBaseline{Mode, Name, M<:AbstractMatrix{Float64}} <: AbstractBaseline
    matrix::M
end
PreparedLegendreBaseline{Mode,Name}(matrix::M) where {Mode,Name,M<:AbstractMatrix{Float64}} =
    PreparedLegendreBaseline{Mode,Name,M}(matrix)

"""
    prepare(baseline, measurement)

`LegendreBaseline`: a `PreparedLegendreBaseline` holding the Legendre matrix on the
measurement's observation axis. `NoBaseline`: unchanged.
"""
prepare(b::LegendreBaseline{Mode,Name}, m::Measurement) where {Mode,Name} =
    PreparedLegendreBaseline{Mode,Name}(legendre_matrix(length(m.y), b.n_coefficients))
prepare(b::NoBaseline, m::Measurement) = b
prepare(b::PreparedLegendreBaseline, m::Measurement) =
    size(b.matrix, 1) == length(m.y) ? b :
    throw(DimensionMismatch("prepared baseline has $(size(b.matrix, 1)) rows, measurement has $(length(m.y))"))

"""
    n_parameters(baseline) -> Int

Number of state entries used by the baseline (0 for `NoBaseline`).

```julia
n_parameters(LegendreBaseline(5))   # 5
```
"""
n_parameters(::NoBaseline) = 0
n_parameters(b::LegendreBaseline) = b.n_coefficients
n_parameters(b::PreparedLegendreBaseline) = size(b.matrix, 2)

"""Name of the `x.baseline` entry a baseline reads (`nothing` for `NoBaseline`)."""
baseline_name(::NoBaseline) = nothing
baseline_name(::LegendreBaseline{M,Name}) where {M,Name} = Name
baseline_name(::PreparedLegendreBaseline{M,Name}) where {M,Name} = Name

@inline baseline_coefficients(::PreparedLegendreBaseline{M,Name}, x) where {M,Name} =
    getproperty(x.baseline, Name)
@inline baseline_term(b::PreparedLegendreBaseline, x) = b.matrix * baseline_coefficients(b, x)

"""
    apply_baseline(prepared_baseline, space, y, x)

Model in measurement space with the baseline applied; `y` is the instrument output already
in `space`, and coefficients are read from `x.baseline.<Name>`. Multiplicative in
`LinearSpace`: `y .* (P * c)`; multiplicative in `LogSpace`: `y .+ P * c`; additive in either
space: `y .+ P * c`; `NoBaseline`: `y`.

```julia
b = prepare(LegendreBaseline(1), measurement)
apply_baseline(b, LinearSpace(), y, ComponentVector(baseline = (window_1 = [2.0],)))   # 2 .* y
```
"""
@inline apply_baseline(::NoBaseline, ::AbstractMeasurementSpace, y::AbstractVector, x) = y
@inline apply_baseline(b::PreparedLegendreBaseline{Multiplicative}, ::LinearSpace,
                       y::AbstractVector, x) = y .* baseline_term(b, x)
@inline apply_baseline(b::PreparedLegendreBaseline{Multiplicative}, ::LogSpace,
                       y::AbstractVector, x) = y .+ baseline_term(b, x)
@inline apply_baseline(b::PreparedLegendreBaseline{Additive}, ::AbstractMeasurementSpace,
                       y::AbstractVector, x) = y .+ baseline_term(b, x)
apply_baseline(::LegendreBaseline, ::AbstractMeasurementSpace, y, x) =
    throw(ArgumentError("apply_baseline needs a prepared baseline; call prepare(baseline, measurement)"))

"""
    initial_coefficients(baseline, space, measurement) -> Vector{Float64}

Starting baseline coefficients. Multiplicative: the maximum of `measurement.y` as the first
coefficient (its log in `LogSpace`) and zeros otherwise, as v0.1 `prior_shape_params`.
Additive: zeros (the signal is not scaled). `NoBaseline`: an empty vector.

```julia
initial_coefficients(LegendreBaseline(3), LinearSpace(), measurement)   # [maximum(y), 0, 0]
```
"""
initial_coefficients(::NoBaseline, ::AbstractMeasurementSpace, m::Measurement) = Float64[]
initial_coefficients(b::Union{LegendreBaseline{Multiplicative},PreparedLegendreBaseline{Multiplicative}},
                     space::AbstractMeasurementSpace, m::Measurement) =
    [first_coefficient(space, maximum(m.y)); zeros(n_parameters(b) - 1)]
initial_coefficients(b::Union{LegendreBaseline{Additive},PreparedLegendreBaseline{Additive}},
                     ::AbstractMeasurementSpace, m::Measurement) = zeros(n_parameters(b))

first_coefficient(::LinearSpace, y_max) = Float64(y_max)
function first_coefficient(::LogSpace, y_max)
    y_max > 0 || throw(ArgumentError("LogSpace needs a positive measurement maximum, got $y_max"))
    return Float64(log(y_max))
end

export legendre_matrix, LegendreBaseline, PreparedLegendreBaseline, baseline_name
