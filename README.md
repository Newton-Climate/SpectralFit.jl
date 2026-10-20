# SpectralFits.jl

Optimal-estimation retrievals of trace gases from spectral measurements: dual-comb
spectroscopy, Fourier transform and grating spectrometers, and open-path
wavelength-modulation spectroscopy, with one forward model and one set of solvers. Every
retrieval reports its posterior covariance, averaging kernel, degrees of freedom for signal,
chi squared and prior influence.

Version 0.2 is a rewrite; it does not provide the v0.1 `Dict`-based API (tag
`v0.1-legacy`). See [ARCHITECTURE.md](ARCHITECTURE.md) for the layout and
[docs/architecture_plan.md](docs/architecture_plan.md) for the design.

## Installation

```julia
using Pkg
Pkg.add(url = "https://github.com/Newton-Climate/SpectralFit.jl.git")
```

## Example: one DCS measurement

```julia
using SpectralFits, Dates

h2o = load_cross_section_model(HitranVoigt, "hit20.par"; molecule = 1, isotope = 1, range_cm = (6075.0, 6079.0))
ch4 = load_cross_section_model(HitranVoigt, "hit20.par"; molecule = 6, isotope = 1, range_cm = (6075.0, 6079.0))
absorbers = (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))

config = RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [5],
                         fine_grid_spacing_cm = 0.003)

measurements = read_measurements(DCS, "20160921.h5")
m = time_average(measurements, Minute(15))[1]

n_air = homogeneous_air_column(m.conditions.pressure_hPa, m.conditions.temperature_K,
                               m.conditions.path_length_cm)
x_prior = ComponentVector(H2O = 0.01n_air, CH4 = 1.9e-6n_air, baseline = (window_1 = zeros(5),))
prior_sigma = ComponentVector(H2O = n_air, CH4 = 1e-6n_air, baseline = (window_1 = fill(1e3, 5),))

result = solve(build_problem(config, absorbers, m, x_prior, prior_sigma), config.solver)
print(report(result))
```

Batch processing of many files is `SpectralFits.Apps.BatchRetrieval` (see section 9 of the
plan).

## Tests and benchmarks

The package targets Julia 1.10+ (developed on 1.12).

```
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
julia --project=test benchmark/forward_model_benchmarks.jl
```

Tests use the HITRAN 2020 subset in `test/data/hit20_subset.par`, so no external line lists
are needed. `test/regression/` compares spectra and retrievals with reference results from
v0.1.
