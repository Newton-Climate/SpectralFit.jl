# Profile retrieval: CH4 and H2O per layer in a 4-layer vertical column (solar-viewing
# geometry), Levenberg-Marquardt, with diagnostics of what the measurement resolves.
#
# Runs without external data (HITRAN 2020 subset in test/data):
#     julia --project=test examples/profile_retrieval.jl

using SpectralFits, Random, LinearAlgebra

par = joinpath(pkgdir(SpectralFits), "test", "data", "hit20_subset.par")
range_cm = (6076.0, 6084.0)
absorbers = (H2O = Absorber(:H2O, load_cross_section_model(HitranVoigt, par; molecule = 1,
                                                           isotope = 1, range_cm = range_cm)),
             CH4 = Absorber(:CH4, load_cross_section_model(HitranVoigt, par; molecule = 6,
                                                           isotope = 1, range_cm = range_cm)))

# Layer-centre pressures and temperatures; layer dry-air columns come from hydrostatic balance.
pressure_hPa = [900.0, 700.0, 500.0, 300.0]
atm = LayeredColumn(pressure_hPa = pressure_hPa, temperature_K = [288.0, 275.0, 255.0, 235.0],
                    solar_zenith_deg = 40.0)
n_layer = hydrostatic_dry_air_column.(layer_pressure_thickness_hPa(pressure_hPa))  # molecules/cm2

# Synthetic measurement from a finer grid than the retrieval uses (avoids an inverse crime).
nu = collect(6078.0:0.0067:6082.0)
noise_sigma = 1e-4
x_true = ComponentVector(H2O = [0.008, 0.004, 0.0015, 0.0004] .* n_layer,
                         CH4 = [1950e-9, 1900e-9, 1850e-9, 1780e-9] .* n_layer,
                         baseline = (window_1 = [1.0, 0.01, -0.005],))
template = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = noise_sigma^2)
truth = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                            ColumnDensity(), template, x_true;
                            wavenumber_fine_cm = 6077.0:0.0005:6083.0)
y = truth(x_true) .+ noise_sigma .* randn(MersenneTwister(7), length(nu))
m = Measurement(y = y, coordinate = nu, noise_variance = noise_sigma^2)

# Forward model and problem assembled by hand (RetrievalConfig covers single-path retrievals).
x_prior = ComponentVector(H2O = 0.004 .* n_layer, CH4 = fill(1800e-9, 4) .* n_layer,
                          baseline = (window_1 = [maximum(y), 0.0, 0.0],))
prior_sigma = ComponentVector(H2O = 0.005 .* n_layer, CH4 = fill(300e-9, 4) .* n_layer,
                              baseline = (window_1 = [1.0, 1.0, 1.0],))
model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                            ColumnDensity(), m, x_prior; wavenumber_fine_cm = 6077.0:0.005:6083.0)
problem = RetrievalProblem(model, observation_vector(LinearSpace(), m),
                           obs_precision(LinearSpace(), m), x_prior,
                           prior_precision_from_sigma(prior_sigma, x_prior))
result = solve(problem, LevenbergMarquardt())
print(report(result))

# The column total is well constrained even though single layers are not.
ch4 = findall(startswith("CH4"), state_labels(result.x))
h = zeros(length(result.x)); h[ch4] .= 1
total, sigma = sum(result.x.CH4), sqrt(h' * result.posterior_covariance * h)
println("\nCH4 column $(round(total / sum(x_true.CH4); digits = 4)) x truth, " *
        "relative sigma $(round(sigma / total; digits = 4))")
println("CH4 DFS per layer: ", round.(result.dfs_per_element.CH4; digits = 2))
