# Digital filters for the WMS lock-in (section 3.7 of docs/architecture_plan.md, open
# question 10: a core Butterworth design instead of a DSP.jl dependency).
#
# Design: analogue Butterworth prototype, cutoff prewarped to 2 fs tan(pi fc / fs), bilinear
# transform z = (2 fs + s) / (2 fs - s), poles paired into second-order sections. Each section
# has unit gain at zero frequency, so the cascade does too. Filtering runs the sections in
# transposed direct form II from a zero initial state and is generic in element type, so
# ForwardDiff dual numbers pass through.

"""
    Butterworth(; order, cutoff_Hz)

Low-pass Butterworth filter specification: `order` (positive integer) and the -3 dB cutoff
frequency `cutoff_Hz` in Hz. Turn it into coefficients for a sampling rate with
`design_filter`.

```julia
lowpass = Butterworth(order = 5, cutoff_Hz = 100.0)
sos = design_filter(lowpass, 20_000.0)
```
"""
struct Butterworth
    order::Int
    cutoff_Hz::Float64
    function Butterworth(order::Integer, cutoff_Hz::Real)
        order >= 1 || throw(ArgumentError("Butterworth: order must be at least 1, got $order"))
        cutoff_Hz > 0 || throw(ArgumentError("Butterworth: cutoff_Hz must be positive, got $cutoff_Hz"))
        return new(Int(order), Float64(cutoff_Hz))
    end
end
Butterworth(; order::Integer, cutoff_Hz::Real) = Butterworth(order, cutoff_Hz)

"""
    SecondOrderSections(b, a, sampling_rate_Hz)

Cascade of second-order IIR sections. Column `k` of `b` (3 rows) holds the numerator
`b0 + b1 z^-1 + b2 z^-2` of section `k`; column `k` of `a` (2 rows) holds `a1, a2` of the
denominator `1 + a1 z^-1 + a2 z^-2`. `sampling_rate_Hz` is the rate the coefficients were
designed for. Built by `design_filter`; applied with `filter_signal`.

```julia
sos = design_filter(Butterworth(order = 2, cutoff_Hz = 50.0), 1000.0)
size(sos.b)   # (3, 1)
```
"""
struct SecondOrderSections
    b::Matrix{Float64}
    a::Matrix{Float64}
    sampling_rate_Hz::Float64
    function SecondOrderSections(b::AbstractMatrix, a::AbstractMatrix, sampling_rate_Hz::Real)
        size(b, 1) == 3 || throw(DimensionMismatch("SecondOrderSections: b must have 3 rows"))
        size(a, 1) == 2 || throw(DimensionMismatch("SecondOrderSections: a must have 2 rows"))
        size(a, 2) == size(b, 2) ||
            throw(DimensionMismatch("SecondOrderSections: b and a need the same number of sections"))
        return new(Matrix{Float64}(b), Matrix{Float64}(a), Float64(sampling_rate_Hz))
    end
end

n_sections(sos::SecondOrderSections) = size(sos.b, 2)

"""
    design_filter(filter::Butterworth, sampling_rate_Hz) -> SecondOrderSections

Digital Butterworth low-pass by bilinear transform with the cutoff prewarped, so the digital
response is exactly -3 dB at `cutoff_Hz`. Requires `cutoff_Hz < sampling_rate_Hz / 2`.
Sections are ordered from the pole farthest from the unit circle to the nearest (a real pole
first for odd orders), each normalized to unit gain at 0 Hz.

```julia
sos = design_filter(Butterworth(order = 5, cutoff_Hz = 100.0), 20_000.0)
abs(frequency_response(sos, 100.0))^2   # 0.5
```
"""
function design_filter(filter::Butterworth, sampling_rate_Hz::Real)
    fs = Float64(sampling_rate_Hz)
    fs > 0 || throw(ArgumentError("design_filter: sampling_rate_Hz must be positive"))
    filter.cutoff_Hz < fs / 2 ||
        throw(ArgumentError("design_filter: cutoff_Hz $(filter.cutoff_Hz) must be below the Nyquist frequency $(fs / 2)"))
    N = filter.order
    warped = 2fs * tan(pi * filter.cutoff_Hz / fs)          # prewarped analogue cutoff, rad/s
    bilinear(s) = (2fs + s) / (2fs - s)
    # analogue prototype poles in the upper half plane, k = 1 .. N/2, and one real pole if N odd
    complex_poles = [bilinear(warped * cis(pi * (2k + N - 1) / (2N))) for k in 1:(N ÷ 2)]
    sort!(complex_poles; by = abs)                          # least resonant section first
    n_sec = (N + 1) ÷ 2
    b = zeros(3, n_sec)
    a = zeros(2, n_sec)
    col = 0
    if isodd(N)
        zr = real(bilinear(-warped))
        col += 1
        g = (1 - zr) / 2
        b[:, col] .= (g, g, 0.0)
        a[:, col] .= (-zr, 0.0)
    end
    for z in complex_poles
        col += 1
        a1 = -2 * real(z)
        a2 = abs2(z)
        g = (1 + a1 + a2) / 4
        b[:, col] .= (g, 2g, g)
        a[:, col] .= (a1, a2)
    end
    return SecondOrderSections(b, a, fs)
end

"""
    frequency_response(sos::SecondOrderSections, frequency_Hz) -> Complex

Complex response `H(exp(i 2 pi f / fs))` of the cascade at `frequency_Hz`.

```julia
sos = design_filter(Butterworth(order = 2, cutoff_Hz = 50.0), 1000.0)
frequency_response(sos, 0.0)   # 1.0 + 0.0im
```
"""
function frequency_response(sos::SecondOrderSections, frequency_Hz::Real)
    zinv = cis(-2pi * frequency_Hz / sos.sampling_rate_Hz)
    h = one(ComplexF64)
    for k in 1:n_sections(sos)
        num = sos.b[1, k] + sos.b[2, k] * zinv + sos.b[3, k] * zinv^2
        den = 1 + sos.a[1, k] * zinv + sos.a[2, k] * zinv^2
        h *= num / den
    end
    return h
end

"""
    transfer_function_coefficients(sos) -> (b, a)

Expanded numerator and denominator polynomials in `z^-1` (`a[1] == 1`), the form returned by
`scipy.signal.butter(order, cutoff, fs = fs)`. For tests and comparison only; filter with the
sections, which are numerically better conditioned.
"""
function transfer_function_coefficients(sos::SecondOrderSections)
    b = [1.0]
    a = [1.0]
    for k in 1:n_sections(sos)
        b = polynomial_product(b, sos.b[:, k])
        a = polynomial_product(a, [1.0, sos.a[1, k], sos.a[2, k]])
    end
    # an odd-order cascade carries one trailing zero coefficient from the first-order section
    order = count(k -> sos.a[2, k] != 0, 1:n_sections(sos)) * 2 +
            count(k -> sos.a[2, k] == 0, 1:n_sections(sos))
    return b[1:order+1], a[1:order+1]
end

function polynomial_product(p::AbstractVector, q::AbstractVector)
    r = zeros(promote_type(eltype(p), eltype(q)), length(p) + length(q) - 1)
    for i in eachindex(p), j in eachindex(q)
        r[i+j-1] += p[i] * q[j]
    end
    return r
end

"""
    filter_signal(sos::SecondOrderSections, signal::AbstractVector) -> Vector

Run `signal` (uniformly sampled at `sos.sampling_rate_Hz`) through the cascade from a zero
initial state, like `scipy.signal.sosfilt`. The output element type is promoted from the input,
so ForwardDiff dual numbers pass through.

```julia
sos = design_filter(Butterworth(order = 5, cutoff_Hz = 100.0), 20_000.0)
y = filter_signal(sos, ones(20_000))   # step response, settles to 1
```
"""
function filter_signal(sos::SecondOrderSections, signal::AbstractVector)
    T = promote_type(eltype(signal), Float64)
    y = Vector{T}(undef, length(signal))
    copyto!(y, signal)
    for k in 1:n_sections(sos)
        b0, b1, b2 = sos.b[1, k], sos.b[2, k], sos.b[3, k]
        a1, a2 = sos.a[1, k], sos.a[2, k]
        s1 = zero(T)
        s2 = zero(T)
        @inbounds for i in eachindex(y)
            xi = y[i]
            yi = b0 * xi + s1
            s1 = b1 * xi - a1 * yi + s2
            s2 = b2 * xi - a2 * yi
            y[i] = yi
        end
    end
    return y
end

export Butterworth, SecondOrderSections, design_filter, frequency_response
export transfer_function_coefficients, filter_signal
