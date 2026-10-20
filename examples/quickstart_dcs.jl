# Quick start: retrieve CH4 and H2O from one synthetic dual-comb (DCS) spectrum.
#
# Runs without external data: it uses the HITRAN 2020 subset bundled in test/data.
#     julia --project=test examples/quickstart_dcs.jl

using SpectralFits, Random

par = joinpath(pkgdir(SpectralFits), "test", "data", "hit20_subset.par")

# 1. Cross-section models (cm2/molecule), one per absorber, over a range wider than the window.
range_cm = (6076.0, 6084.0)
h2o = load_cross_section_model(HitranVoigt, par; molecule = 1, isotope = 1, range_cm = range_cm)
ch4 = load_cross_section_model(HitranVoigt, par; molecule = 6, isotope = 1, range_cm = range_cm)
absorbers = (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))

# 2. A synthetic measurement. Real data come from `read_measurements` (see README).
conditions = (pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
n_air = homogeneous_air_column(conditions.pressure_hPa, conditions.temperature_K,
                               conditions.path_length_cm)            # molecules/cm2
nu = collect(6077.0:0.0067:6083.0)                                     # cm-1
noise_sigma = 2e-4
template = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = noise_sigma^2,
                       conditions = conditions)
x_true = ComponentVector(H2O = 0.012 * n_air, CH4 = 1950e-9 * n_air,
                         baseline = (window_1 = [1.0, 0.02, -0.01],))
config = RetrievalConfig(windows_cm = [(6077.5, 6082.5)], baseline_coefficients = [3],
                         fine_grid_spacing_cm = 0.002)
truth = build_forward_model(config, absorbers, template, x_true)
y = truth(x_true) .+ noise_sigma .* randn(MersenneTwister(1), length(truth(x_true)))
window = select_window(template, 6077.5, 6082.5)
measurement = Measurement(y = y, coordinate = window.coordinate, noise_variance = noise_sigma^2,
                          conditions = conditions)

# 3. Prior: state entries are named after the absorbers; columns in molecules/cm2.
x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1800e-9 * n_air,
                          baseline = (window_1 = zeros(3),))   # set from the measurement
prior_sigma = ComponentVector(H2O = 0.01 * n_air, CH4 = 500e-9 * n_air,
                              baseline = (window_1 = fill(10.0, 3),))

# 4. Build, solve, report.
problem = build_problem(config, absorbers, measurement, x_prior, prior_sigma)
result = solve(problem, config.solver)
print(report(result))

xch4_ppb = 1e9 * result.x.CH4 / n_air
sigma_ppb = 1e9 * sqrt(result.posterior_covariance[2, 2]) / n_air
println("\nCH4 = $(round(xch4_ppb; digits = 1)) +/- $(round(sigma_ppb; digits = 1)) ppb (truth 1950)")

# 5. Save the result (plain data, loads with JLD2 alone).
path = save_results(joinpath(mktempdir(), "quickstart_results.jld2"), [result])
println("saved to $path")
