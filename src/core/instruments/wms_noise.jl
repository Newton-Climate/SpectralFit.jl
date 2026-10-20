# WMS observation noise covariance (section 8, open question 9 of docs/architecture_plan.md):
# the lock-in low-pass correlates neighbouring output samples, so the observation covariance
# is banded rather than diagonal. Used with `CovariancePrecision`, which applies its inverse by
# sparse Cholesky solves.
#
# Model. The detector adds white noise e_m of variance sigma2 to every raw lock-in sample. The
# quadrature X_n = LP[2 S cos(n w t)] carries the noise u_m = 2 cos(n w t_m) e_m, with
# E[u_m^2] = 4 cos^2(n w t_m) sigma2 = 2 sigma2 (1 + cos(2 n w t_m)). With impulse response h of
# the low-pass (in samples) the filtered noise has
#     Cov(X_n[i], X_n[j]) = 2 sigma2 sum_m h[i-m] h[j-m] (1 + cos(2 n w t_m))
#                         ~ 2 sigma2 R_h[i - j],    R_h[d] = sum_m h[m] h[m + d],
# dropping the term at 2 n f, which the low-pass suppresses. The mixing gain is g = 2 for X_n
# and Y_n (mean of (2 cos)^2 and (2 sin)^2) and g = 1 for DC (no mixing). For the same reason
# (products of references at frequencies k f, k >= 1, well above the cutoff) X_n, Y_n, X_m and
# DC are mutually uncorrelated to the same order. Observation times between lock-in samples
# use the same linear interpolation as `demodulate`, so the covariance is propagated through
# the interpolation weights. The filter is assumed to have settled (steady state); use
# `pre_roll_s` of several filter time constants, as for the signal.

# Moduli of the poles of the cascade, to size the impulse response.
function largest_pole_modulus(sos::SecondOrderSections)
    r = 0.0
    for k in 1:n_sections(sos)
        a1, a2 = sos.a[1, k], sos.a[2, k]
        disc = a1^2 - 4a2
        rk = disc < 0 ? sqrt(a2) : max(abs((-a1 + sqrt(disc)) / 2), abs((-a1 - sqrt(disc)) / 2))
        r = max(r, rk)
    end
    return r
end

"""
    lowpass_autocorrelation(sos::SecondOrderSections) -> Vector{Float64}

Autocorrelation `R[d + 1] = sum_m h[m] h[m + d]` (d = 0, 1, ... samples) of the impulse
response `h` of the cascade, so that white noise of unit variance per sample leaves the filter
with covariance `R[|i - j| + 1]` between output samples i and j (steady state). The impulse
response is run until its envelope has decayed below 1e-22 and trimmed where the remaining
energy is negligible. Computed by filtering the time-reversed impulse response, with no FFT.

```julia
sos = design_filter(Butterworth(order = 4, cutoff_Hz = 400.0), 100e3)
R = lowpass_autocorrelation(sos)
R[1]   # noise-equivalent bandwidth over half the sampling rate
```
"""
function lowpass_autocorrelation(sos::SecondOrderSections)
    r = largest_pole_modulus(sos)
    r < 1 || throw(ArgumentError("sos: the filter is not stable (pole modulus $r)"))
    n_poles = 2 * n_sections(sos)
    L = ceil(Int, 50 / -log(r)) * (1 + n_poles ÷ 2) + 16
    impulse = zeros(L)
    impulse[1] = 1.0
    h = filter_signal(sos, impulse)
    tail = reverse(cumsum(reverse(h .^ 2)))          # energy from sample m to the end
    last_kept = something(findlast(>(1e-20 * tail[1]), tail), 1)
    h = h[1:last_kept]
    return reverse(filter_signal(sos, reverse(h)))   # R[d + 1] for d = 0 .. last_kept - 1
end

"""
    lockin_output_covariance(lockin::DigitalLockIn, times_s; band_tolerance = 1e-6)
        -> SparseMatrixCSC{Float64}

Covariance of one low-pass output of `lockin` at the measurement times `times_s` (s) for
white noise of unit variance per raw sample entering the filter (before any mixing gain):
`C[k, l] = sum_ab w_a(k) w_b(l) R[|i_k + a - i_l - b|]`, with `R` the filter autocorrelation
(`lowpass_autocorrelation`) at the lag in samples and `w` the linear interpolation weights of
`demodulate` on the sample grid of `lockin_sample_times`. Lags where `|R| / R(0)` has fallen
below `band_tolerance` for good are set to zero, so the matrix is banded (symmetric, sparse).
Times after the first need not be sorted (the first anchors the sample grid, as in `prepare`).

```julia
lockin = DigitalLockIn(lowpass = Butterworth(order = 4, cutoff_Hz = 400.0), sampling_rate_Hz = 100e3)
C = lockin_output_covariance(lockin, range(0.01, 0.19; length = 90))
```
"""
function lockin_output_covariance(lockin::DigitalLockIn, times_s::AbstractVector{<:Real};
                                  band_tolerance::Real = 1e-6)
    0 < band_tolerance < 1 ||
        throw(ArgumentError("band_tolerance: must be in (0, 1), got $band_tolerance"))
    isempty(times_s) && throw(ArgumentError("times_s: is empty"))
    _, index, weight = lockin_sampling(lockin, collect(Float64, times_s))
    R = lowpass_autocorrelation(design_filter(lockin.lowpass, lockin.sampling_rate_Hz))
    d_max = something(findlast(v -> abs(v) >= band_tolerance * R[1], R), 1) - 1
    R_lag(d) = d <= d_max ? R[d+1] : 0.0
    K = length(index)
    order = sortperm(index)
    rows, cols, values = Int[], Int[], Float64[]
    for a in 1:K
        k = order[a]
        wk = (1 - weight[k], weight[k])
        for b in a:K
            l = order[b]
            index[l] - index[k] > d_max + 1 && break
            wl = (1 - weight[l], weight[l])
            c = 0.0
            for p in 0:1, q in 0:1
                c += wk[p+1] * wl[q+1] * R_lag(abs(index[k] + p - index[l] - q))
            end
            c == 0 && continue
            push!(rows, k); push!(cols, l); push!(values, c)
            k == l || (push!(rows, l); push!(cols, k); push!(values, c))
        end
    end
    return sparse(rows, cols, values, K, K)
end

# Components of a HarmonicSignals: X of each harmonic, then Y of each, then DC; mixing gains.
n_signal_components(h::HarmonicSignals{N}) where {N} = 2N + 1
mixing_gain(h::HarmonicSignals{N}, c::Integer) where {N} = c <= 2N ? 2.0 : 1.0

function perturb_component(h::HarmonicSignals{N}, c::Integer, eps) where {N}
    T = typeof(eps)
    X = Matrix{T}(h.X)
    Y = Matrix{T}(h.Y)
    dc = Vector{T}(h.dc)
    if c <= N
        X[:, c] .+= eps
    elseif c <= 2N
        Y[:, c-N] .+= eps
    else
        dc .+= eps
    end
    return HarmonicSignals(h.harmonics, X, Y, dc)
end

"""
    observable_signal_jacobian(observable, signals::HarmonicSignals, x) -> Matrix{Float64}

Derivatives of the observable values (rows, `n_obs = K * observation_multiplicity`) with
respect to each demodulated component at the same slow-scan time (columns: `X_n` for each
harmonic of `signals`, then `Y_n`, then DC), by ForwardDiff at `signals`. Assumes, as for
every observable in this package, that an observation at time k depends only on the
components at time k. For a `BackgroundSubtracted` observable the background does not carry
detector noise, so the inner observable is used.
"""
function observable_signal_jacobian(observable, signals::HarmonicSignals, x)
    inner = observable isa BackgroundSubtracted ? observable.observable : observable
    J = reduce(hcat, [ForwardDiff.derivative(e -> evaluate_observable(inner, perturb_component(signals, c, e), x), 0.0)
                      for c in 1:n_signal_components(signals)])
    return Matrix{Float64}(J)
end

linear_in_signals(::AbstractWMSObservable) = false
linear_in_signals(::Quadratures) = true
linear_in_signals(o::BackgroundSubtracted) = linear_in_signals(o.observable)

"""
    wms_noise_covariance(prepared::PreparedWavelengthModulation, lockin::DigitalLockIn,
                         detector_variance; signals = nothing, x = nothing,
                         band_tolerance = 1e-6) -> SparseMatrixCSC{Float64}
    wms_noise_covariance(prepared, detector_variance; kwargs...)

Banded observation covariance (`n_obs(prepared) x n_obs(prepared)`, observable units squared)
of a WMS measurement whose detector adds white noise of variance `detector_variance`
(detector units squared) to every raw sample of the digital lock-in `lockin`. The second form
uses the prepared instrument's own `DigitalLockIn`; the first lets a retrieval with the
quasi-static `HarmonicExpansion` describe data recorded through a real lock-in. Wrap the
result in `CovariancePrecision` for `RetrievalProblem`.

Each demodulated component has covariance `g * detector_variance * C`, with `C` from
`lockin_output_covariance` (the filter autocorrelation at the lag between observation times,
banded at `band_tolerance`) and mixing gain `g = 2` for `X_n`, `Y_n` (references `2 cos`,
`2 sin`) and `g = 1` for DC; distinct components are uncorrelated. This drops terms at twice
the harmonic frequencies, so it needs the low-pass cutoff well below the modulation frequency
(checked: `cutoff_Hz < frequency_Hz / 2`).

Observables are propagated to first order, `Cov(o) = sum_c g_c D_c (detector_variance C) D_c`
with `D_c` the diagonal of derivatives of the observable with respect to component c at the
same time (`observable_signal_jacobian`):

- `Quadratures`: exact (derivatives 0 or 1); `X_n` and `Y_n` blocks are uncorrelated.
- `HarmonicRatio`, `HarmonicOverDC`, `HarmonicProjection`, `BackgroundSubtracted`:
  linearised about `signals`, the noise-free `HarmonicSignals` at the measurement times
  (`harmonic_signals(prepared, grid, transmission, x)` at a first guess, or
  `demodulate(prepared, trace)` of data), which is then required. The linearisation is good
  while every normalizing magnitude or DC level is far above its noise; magnitudes near zero
  (Rician bias) are not modelled. `x` resolves state-dependent observable slots (projection
  phase); the background of `BackgroundSubtracted` is taken as noise free.

Throws an `ArgumentError` naming the argument for a non-positive `detector_variance`, a
missing or mis-sized `signals`, non-finite derivatives, or a cutoff too close to the
modulation frequency.

```julia
retrieval = prepare(WavelengthModulation(tuning, laser, HarmonicExpansion(), Quadratures((2,))), m)
lockin = DigitalLockIn(lowpass = Butterworth(order = 4, cutoff_Hz = 400.0), sampling_rate_Hz = 100e3)
Se = wms_noise_covariance(retrieval, lockin, 1e-6)
problem = RetrievalProblem(model, m.y, CovariancePrecision(Se), x_prior, Sa_inv)
```
"""
function wms_noise_covariance(prepared::PreparedWavelengthModulation, lockin::DigitalLockIn,
                              detector_variance::Real; signals = nothing, x = nothing,
                              band_tolerance::Real = 1e-6)
    (isfinite(detector_variance) && detector_variance > 0) ||
        throw(ArgumentError("detector_variance: must be positive and finite, got $detector_variance"))
    w = prepared.instrument
    f_mod = w.tuning.modulation.frequency_Hz
    lockin.lowpass.cutoff_Hz < f_mod / 2 || throw(ArgumentError(
        "lockin: low-pass cutoff_Hz $(lockin.lowpass.cutoff_Hz) must be below half the modulation " *
        "frequency $f_mod Hz for the banded noise model"))
    observable = w.observable
    times = prepared.times_s
    K = length(times)
    if signals === nothing
        linear_in_signals(observable) || throw(ArgumentError(
            "signals: needed to linearise the noise of $(nameof(typeof(observable))); pass the " *
            "noise-free HarmonicSignals at the measurement times"))
        harmonics = required_harmonics(observable)
        N = length(harmonics)
        signals = HarmonicSignals(harmonics, ones(K, N), ones(K, N), ones(K))
    end
    signals isa HarmonicSignals ||
        throw(ArgumentError("signals: must be a HarmonicSignals, got $(typeof(signals))"))
    length(signals.dc) == K || throw(ArgumentError(
        "signals: has $(length(signals.dc)) times but the measurement has $K"))
    J = observable_signal_jacobian(observable, signals, x)
    n_obs = n_observations(prepared)
    size(J, 1) == n_obs ||
        throw(ArgumentError("signals: the observable gives $(size(J, 1)) values, expected $n_obs"))
    all(isfinite, J) || throw(ArgumentError(
        "signals: the observable's derivatives are not finite (a normalizing magnitude or DC " *
        "level is zero)"))
    gains = [mixing_gain(signals, c) for c in axes(J, 2)]
    C = lockin_output_covariance(lockin, times; band_tolerance = band_tolerance)
    m = n_obs ÷ K
    rows, cols, values = Int[], Int[], Float64[]
    Ck, Cl, Cv = findnz(C)
    for b1 in 1:m, b2 in 1:m, (k, l, v) in zip(Ck, Cl, Cv)
        i = (b1 - 1) * K + k
        j = (b2 - 1) * K + l
        s = 0.0
        for c in eachindex(gains)
            s += gains[c] * J[i, c] * J[j, c]
        end
        s == 0 && continue
        push!(rows, i); push!(cols, j); push!(values, detector_variance * v * s)
    end
    return sparse(rows, cols, values, n_obs, n_obs)
end

function wms_noise_covariance(prepared::PreparedWavelengthModulation, detector_variance::Real; kwargs...)
    lockin = prepared.instrument.demodulator
    lockin isa DigitalLockIn || throw(ArgumentError(
        "prepared: its demodulator is not a DigitalLockIn; pass the lock-in that recorded the " *
        "data, wms_noise_covariance(prepared, lockin, detector_variance)"))
    return wms_noise_covariance(prepared, lockin, detector_variance; kwargs...)
end

export lowpass_autocorrelation, lockin_output_covariance, observable_signal_jacobian
export wms_noise_covariance
