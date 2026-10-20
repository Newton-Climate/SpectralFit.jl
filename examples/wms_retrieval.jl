# Open-path wavelength-modulation spectroscopy (WMS): retrieve CH4 from 2f quadratures, with
# the banded noise covariance of a digital lock-in.
#
# The absorber is a single analytic Lorentz line, defined below in four lines: any type with
# `cross_section`, `depends_on` and `spectral_range_cm` methods can be an absorber.
#     julia --project=test examples/wms_retrieval.jl

using SpectralFits, Random
import SpectralFits: cross_section, depends_on, spectral_range_cm

struct LorentzLine <: AbstractCrossSectionModel
    center_cm::Float64          # cm-1
    strength::Float64           # cm/molecule
    hwhm_cm_per_hPa::Float64    # pressure-broadened half width
end
function cross_section(m::LorentzLine, nu, c::GasConditions)
    g = m.hwhm_cm_per_hPa * c.pressure_hPa
    return @. m.strength * (g / pi) / ((nu - m.center_cm)^2 + g^2)
end
depends_on(::LorentzLine) = (:pressure,)
spectral_range_cm(::LorentzLine) = (-Inf, Inf)

nu0, gamma = 6046.95, 0.05
absorbers = (CH4 = Absorber(:CH4, LorentzLine(nu0, 1.6e-20, gamma / 1000)),)
atm = HomogeneousPath(pressure_hPa = 1000.0, temperature_K = 295.0, path_length_cm = 1.0e4)
n_air = dry_air_column(atm)

# Laser: slow ramp across the line plus 5 kHz sinusoidal modulation; intensity modulation
# amplitude is retrieved (state entry instrument.linear_amplitude).
tuning = LaserTuning(LinearRamp(start_cm = nu0 - 0.3, rate_cm_per_s = 3.0),
                     SinusoidalModulation(amplitude_cm = 2.2 * gamma, frequency_Hz = 5e3))
laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1)
observable = Quadratures((2,))                         # X2 and Y2 at each time
times = collect(range(0.02, 0.18; length = 321))       # s; observations are X2 then Y2
m0 = Measurement(y = zeros(2 * length(times)), coordinate = repeat(times, 2), noise_variance = 1.0)

# Retrieval model: quasi-static harmonic expansion (fast, smooth for ForwardDiff).
instrument = WavelengthModulation(tuning, laser, HarmonicExpansion(harmonics = (2,)), observable)
grid = range(nu0 - 0.6, nu0 + 0.6; step = 0.002)
x_true = ComponentVector(CH4 = 2.0e-6 * n_air, instrument = (linear_amplitude = 0.1,))
model = build_forward_model(atm, absorbers, instrument, NoBaseline(), LinearSpace(),
                            ColumnDensity(), m0, x_true; wavenumber_fine_cm = grid)

# Noise: white detector noise demodulated by a lock-in with a 400 Hz Butterworth low-pass.
# The filter correlates neighbouring outputs; wms_noise_covariance returns the banded
# covariance and CovariancePrecision uses it through its sparse Cholesky factor.
lockin = DigitalLockIn(harmonics = (2,), lowpass = Butterworth(order = 4, cutoff_Hz = 400.0),
                       sampling_rate_Hz = 100e3, pre_roll_s = 0.02)
detector_variance = 1e-6
Se = wms_noise_covariance(model.instrument, lockin, detector_variance)

# Synthetic data with that noise, made by running the lock-in on detector noise.
noisy = prepare(WavelengthModulation(tuning, laser, lockin, observable), m0)
detector_noise = sqrt(detector_variance) .* randn(MersenneTwister(3), length(lockin_sample_times(noisy)))
y = model(x_true) .+ evaluate_observable(observable, demodulate(noisy, detector_noise), nothing)

x_prior = ComponentVector(CH4 = 1.6e-6 * n_air, instrument = (linear_amplitude = 0.12,))
prior_sigma = ComponentVector(CH4 = 2.0e-5 * n_air, instrument = (linear_amplitude = 0.5,))
problem = RetrievalProblem(model, y, CovariancePrecision(Se), x_prior,
                           prior_precision_from_sigma(prior_sigma, x_prior))
result = solve(problem, LevenbergMarquardt())
print(report(result))
println("\nCH4 = $(round(1e6 * result.x.CH4 / n_air; digits = 4)) +/- " *
        "$(round(1e6 * sqrt(result.posterior_covariance[1, 1]) / n_air; digits = 4)) ppm (truth 2)")
