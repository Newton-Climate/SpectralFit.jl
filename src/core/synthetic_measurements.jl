# Synthetic measurements for observing system simulation experiments (OSSEs).

"""
    synthetic_measurement(model, x_true, template::Measurement, space, noise_sigma; rng)
        -> Measurement

Noisy synthetic measurement on the coordinate of `template`: the model at `x_true` in
instrument units plus Gaussian noise of standard deviation `noise_sigma` (instrument units,
a scalar or one value per observation). `model(x)` returns its output in measurement `space`;
for `LogSpace()` it is exponentiated before the noise is added, so the noise is always in
intensity, as for a real spectrometer, and `obs_precision(space, m)` gives the matching
weights. `time`, `conditions` and `metadata` are copied from `template`.

Generating the truth with the same model that is used in the retrieval is an inverse crime;
use a finer fine grid or another cross-section model for the truth where that matters.

```julia
m = synthetic_measurement(truth_model, x_true, template, LogSpace(), 1e-3; rng = MersenneTwister(1))
```
"""
function synthetic_measurement(model, x_true, template::Measurement, space::AbstractMeasurementSpace,
                               noise_sigma; rng::AbstractRNG = Random.default_rng())
    signal = _instrument_units(space, model(x_true))
    length(signal) == length(template.y) || throw(DimensionMismatch(
        "synthetic_measurement: model gives $(length(signal)) values, the template has $(length(template.y))"))
    sigma = noise_sigma isa Real ? fill(Float64(noise_sigma), length(signal)) : collect(Float64, noise_sigma)
    length(sigma) == length(signal) || throw(DimensionMismatch(
        "noise_sigma: has $(length(sigma)) values for $(length(signal)) observations"))
    all(>(0), sigma) || throw(ArgumentError("noise_sigma: must be positive"))
    y = signal .+ sigma .* randn(rng, length(signal))
    space isa LogSpace && !all(>(0), y) && throw(ArgumentError(
        "synthetic_measurement: the noisy spectrum has non-positive values, which LogSpace cannot use"))
    return Measurement(y = y, coordinate = template.coordinate, noise_variance = sigma .^ 2,
                       time = template.time, conditions = template.conditions,
                       metadata = template.metadata)
end

_instrument_units(::LinearSpace, y) = collect(Float64, y)
_instrument_units(::LogSpace, y) = exp.(collect(Float64, y))

export synthetic_measurement
