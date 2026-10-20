# SpectralFits.jl

Optimal-estimation retrievals of trace gases from spectral measurements: dual-comb
spectroscopy (DCS), Fourier transform (FTS) and grating spectrometers, and open-path
wavelength-modulation spectroscopy (WMS). One forward model and one set of solvers serve all
of them. Every retrieval reports its posterior covariance, averaging kernel, degrees of
freedom for signal (DFS), chi squared and prior influence.

Version 0.2 is a rewrite and does not provide the v0.1 `Dict`-based API. The old code is at
the tag `v0.1-legacy`; [Coming from v0.1](#coming-from-v01) maps the old names to the new.
[ARCHITECTURE.md](ARCHITECTURE.md) describes the code layout, and
[docs/architecture_plan.md](docs/architecture_plan.md) the design and its decisions.

- [Installation](#installation)
- [Quick start](#quick-start)
- [Concepts](#concepts)
- [Retrieving from DCS files](#retrieving-from-dcs-files)
- [Batch processing](#batch-processing)
- [Configuration options](#configuration-options)
- [Profile retrievals](#profile-retrievals)
- [Other instruments: FTS, grating, WMS](#other-instruments-fts-grating-wms)
- [Results and diagnostics](#results-and-diagnostics)
- [Lookup tables](#lookup-tables)
- [Adding a cross-section model](#adding-a-cross-section-model)
- [Coming from v0.1](#coming-from-v01)
- [Tests, examples and benchmarks](#tests-examples-and-benchmarks)

## Installation

SpectralFits needs Julia 1.10 or later (it is developed on 1.12). It is not in the General
registry, so install it from GitHub:

```julia
using Pkg
Pkg.add(url = "https://github.com/Newton-Climate/SpectralFit.jl.git")
```

To work on the package itself, clone it and use the test environment, which loads the local
source:

```bash
git clone https://github.com/Newton-Climate/SpectralFit.jl.git
```

```bash
julia --project=SpectralFit.jl/test -e 'using Pkg; Pkg.instantiate()'
```

`using SpectralFits` exports the whole public API, including `ComponentVector`, which is used
for state vectors.

## Quick start

This retrieves H2O and CH4 from a synthetic DCS spectrum. It uses the HITRAN 2020 subset that
ships with the tests, so it runs without any data files. The full script is
[examples/quickstart_dcs.jl](examples/quickstart_dcs.jl).

```julia
using SpectralFits

par = joinpath(pkgdir(SpectralFits), "test", "data", "hit20_subset.par")

# cross-section models (cm2/molecule), over a range wider than the retrieval window
h2o = load_cross_section_model(HitranVoigt, par; molecule = 1, isotope = 1, range_cm = (6076.0, 6084.0))
ch4 = load_cross_section_model(HitranVoigt, par; molecule = 6, isotope = 1, range_cm = (6076.0, 6084.0))
absorbers = (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))

# windows, baseline order and fine-grid spacing; the defaults give a DCS instrument,
# linear (intensity) space, column-density units and a Gauss-Newton solver
config = RetrievalConfig(windows_cm = [(6077.5, 6082.5)], baseline_coefficients = [3],
                         fine_grid_spacing_cm = 0.002)

# `measurement` is a Measurement with conditions (pressure_hPa, temperature_K, path_length_cm);
# the script makes a synthetic one. Columns are in molecules/cm2.
c = measurement.conditions
n_air = homogeneous_air_column(c.pressure_hPa, c.temperature_K, c.path_length_cm)
x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1800e-9 * n_air,
                          baseline = (window_1 = zeros(3),))
prior_sigma = ComponentVector(H2O = 0.01 * n_air, CH4 = 500e-9 * n_air,
                              baseline = (window_1 = fill(10.0, 3),))

problem = build_problem(config, absorbers, measurement, x_prior, prior_sigma)
result = solve(problem, config.solver)
print(report(result))
1e9 * result.x.CH4 / n_air        # CH4 dry mole fraction in ppb
```

```bash
julia --project=test examples/quickstart_dcs.jl
```

The report gives the status, iterations, DFS, chi squared, then one CSV row per state
element (prior, retrieved value, posterior sigma, DFS). For this example the retrieval
converges in 3 iterations with chi2 per degree of freedom near 1 and recovers CH4 within its
posterior sigma.

## Concepts

**Measurement.** A `Measurement` holds `y`, its `coordinate` (wavenumber in cm-1 for spectra,
time in s for WMS), `noise_variance`, an optional `time`, `conditions` (for example
`pressure_hPa`, `temperature_K`, `path_length_cm`) and free-form `metadata`. It holds no
physics settings. `select_window(m, lo, hi)` keeps the closed interval `lo <= nu <= hi`, and
`time_average(ms, Minute(15))` averages a series into time bins.

**State vector.** The state is a `ComponentVector` with named entries:

| Entry | Meaning |
|---|---|
| one per absorber (`H2O`, `CH4`, ...) | column (molecules/cm2) or dry mole fraction, one value per layer |
| `pressure_hPa`, `temperature_K` | present only when retrieved |
| `baseline.window_1`, `baseline.window_2`, ... | Legendre coefficients per window |
| `instrument.stretch`, `instrument.linear_amplitude`, ... | retrieved instrument parameters |

The prior `x_prior` and `prior_sigma` have the same entries, so names (not positions) say
what is retrieved. Anything that can be retrieved is a *parameter slot*: give it a number in
the constructor and it is fixed, or add its entry to the state and the builder reads it from
there.

**Forward model.** The model runs optical depth on a fine wavenumber grid, then transmission,
the instrument, the measurement space (linear or log) and the baseline. `build_forward_model`
does all the set-up once: it chooses and checks the fine grid, precomputes cross sections
that do not depend on the state, and checks that the state matches the absorbers and
baseline. The result is a callable `model(x)`, differentiable with ForwardDiff.

**Problem and solvers.** `RetrievalProblem(model, y, obs_precision, x_prior, prior_precision)`
holds the inverse problem; `build_problem(config, ...)` assembles one from a
`RetrievalConfig`. `solve(problem, GaussNewton())` or
`solve(problem, LevenbergMarquardt())` returns a `RetrievalResult` with all diagnostics.
Convergence uses the Rodgers d2 test.

**Layers.** `SpectralFits.CoreLayer` is the science with no file I/O,
`SpectralFits.IOLayer` reads and writes files, and `SpectralFits.Apps.BatchRetrieval` is the
batch workflow.

## Retrieving from DCS files

`read_measurements(DCS, path; reader_config)` reads an HDF5 file from the DCS acquisition
software into one `Measurement` per record. It recognises the three file layouts used so
far; `DCSFieldLayout` names the datasets of any other.

```julia
using SpectralFits, Dates

reader = DCSReaderConfig(
    path_length_cm = 195017.0,   # DCS A round trip (1.95 km)
    pressure_scale = 0.9938,     # 2016 campaign pressure correction; default 1.0
    noise_model = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)],
                                polynomial_order = 3))
records = read_measurements(DCS, "20160921.h5"; reader_config = reader)
averaged = time_average(records, Minute(15))

result = solve(build_problem(config, absorbers, averaged[1], x_prior, prior_sigma), config.solver)
```

Things to know about the reader:

- **Noise** is estimated per record on the full spectrum, before averaging and windowing.
  `ResidualNoise` fits a polynomial to flat (absorption-free) regions near the window and uses
  the residual variance; `ShotNoise()` (the default) is the v0.1 model and needs the flat
  region 6255.1-6255.4 cm-1; `ConstantNoise(v)` sets it directly. Averaging divides the
  variance by the number of records.
- **Times** are seconds since the LabVIEW epoch, `DateTime(1904, 1, 1)`, by default. v0.1
  read them as Unix time, so its timestamps are 66 years late. Pass
  `time_epoch = DateTime(1970, 1, 1)` for files that store Unix time.
- **Invalid records** (non-finite intensity or noise) are skipped with a warning; pass
  `invalid_records = :error` to fail instead.
- **Conditions** are the recorded pressure times `pressure_scale`, the temperature in K and
  the configured path length. `RetrievalConfig` builds the atmosphere from them, so
  pressure and temperature are not in the state unless you retrieve them.

## Batch processing

`SpectralFits.Apps.BatchRetrieval` reads many files, averages them, retrieves every averaged
measurement and writes the results with a manifest. The worked example is
[examples/dcs_merlin_batch.jl](examples/dcs_merlin_batch.jl).

```julia
using SpectralFits, Dates
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, run

config = BatchRetrievalConfig(
    experiment = "merlin-r6",
    data_directory = "../data", file_pattern = r"^2016.*\.h5$",
    reader = DCSReaderConfig(path_length_cm = 195017.0, pressure_scale = 0.9938),
    noise = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)]),
    averaging_period = Minute(15),
    absorbers = absorbers, retrieval = config,
    x_prior = x_prior, prior_sigma = prior_sigma,
    output_root = "../data/runs")

summary = run(config)
print(report(summary))
```

`run` creates `<output_root>/<yyyy-mm-dd>_<experiment>_<git hash>/` with
`configuration.txt`, `manifest.toml` and `results/<input>_results.jld2` (one per input file;
read them with `load_results`). A failed retrieval or unreadable file is logged and recorded,
and the batch continues.

Retrievals run on 2 worker processes by default. `run` starts any missing workers in the
current project with SpectralFits loaded, and removes them when it finishes. Set
`n_workers = 8` for more, or `parallel = :serial` for none. The serial and parallel results
are identical.

## Configuration options

`RetrievalConfig` keywords (only `windows_cm` and `baseline_coefficients` are required):

| Keyword | Default | Meaning |
|---|---|---|
| `windows_cm` | | vector of `(lo, hi)` windows in cm-1, closed intervals |
| `baseline_coefficients` | | Legendre coefficients per window, 0 for no baseline |
| `baseline_mode` | `:multiplicative` | or `:additive` |
| `measurement_space` | `LinearSpace()` | or `LogSpace()` (log transmission, additive baseline) |
| `state_units` | `ColumnDensity()` | or `VMR()` (dry-air mole fractions) |
| `instrument` | `DCS()` | `DCS(calibration = LinearStretch())` retrieves a wavenumber stretch |
| `retrieve_pressure`, `retrieve_temperature` | `false` | must match `pressure_hPa`, `temperature_K` in the state |
| `h2o_broadening` | none | `Dict(:CH4 => 0.34)` adds H2O broadening to that absorber |
| `fine_grid_spacing_cm` | instrument default | fine-grid spacing in cm-1 |
| `masked_windows_cm`, `mask_mode`, `mask_factor` | none, `:deweight`, `1e-5` | de-weight (or `:drop`) points in these windows |
| `baseline_prior` | `:from_measurement` | baseline prior from each measurement, or `:as_given` |
| `solver` | `GaussNewton()` | or `LevenbergMarquardt(damping = AdaptiveDamping())` |

Examples:

```julia
# two windows, each with its own baseline (state entries baseline.window_1, baseline.window_2)
RetrievalConfig(windows_cm = [(6076.05, 6077.95), (6080.0, 6082.0)],
                baseline_coefficients = [5, 3])

# retrieve temperature too: add the entry to the prior and say so
x_prior = ComponentVector(H2O = ..., CH4 = ..., temperature_K = 290.0, baseline = ...)
RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [5],
                retrieve_temperature = true)

# dry mole fractions instead of columns; prior entries are then in mol/mol
RetrievalConfig(windows_cm = [(6076.05, 6077.95)], baseline_coefficients = [5],
                state_units = VMR())
```

For unconstrained least squares as in v0.1, build the problem with `RetrievalProblem` and a
prior precision that is zero for those elements, for example
`Diagonal(zeros(length(x_prior)))`. The diagnostics stay well defined: the averaging kernel
is the identity and the prior influence is 0.

## Profile retrievals

`LayeredColumn` is a vertical column seen at a solar zenith angle. Gas entries hold one value
per layer, and layer dry-air columns come from hydrostatic balance unless you give them. For
a layered atmosphere, call `build_forward_model` directly. The full script is
[examples/profile_retrieval.jl](examples/profile_retrieval.jl).

```julia
pressure_hPa = [900.0, 700.0, 500.0, 300.0]
atm = LayeredColumn(pressure_hPa = pressure_hPa, temperature_K = [288.0, 275.0, 255.0, 235.0],
                    solar_zenith_deg = 40.0)
n_layer = hydrostatic_dry_air_column.(layer_pressure_thickness_hPa(pressure_hPa))

x_prior = ComponentVector(H2O = 0.004 .* n_layer, CH4 = fill(1800e-9, 4) .* n_layer,
                          baseline = (window_1 = [maximum(m.y), 0.0, 0.0],))
model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                            ColumnDensity(), m, x_prior; wavenumber_fine_cm = 6077.0:0.005:6083.0)
problem = RetrievalProblem(model, observation_vector(LinearSpace(), m),
                           obs_precision(LinearSpace(), m), x_prior,
                           prior_precision_from_sigma(prior_sigma, x_prior))
result = solve(problem, LevenbergMarquardt())
result.dfs_per_element.CH4     # what each layer contributes
```

In that example single layers have DFS of 0.3-0.7, while the total CH4 column is retrieved to
about 1%. A temperature profile is retrieved by adding `temperature_K` (one value per layer)
to the state.

## Other instruments: FTS, grating, WMS

Instruments are swapped in the same builder call:

```julia
FTS(max_path_difference_cm = 5.0, apodization = Triangular())   # sinc-type ILS on a sparse matrix
GratingSpectrometer(offsets_cm = offsets, response = response)  # tabulated line shape
DCS(calibration = LinearStretch())                              # retrieved stretch
```

**WMS.** `WavelengthModulation(tuning, laser, demodulator, observable)` models a laser scanned
across a line with sinusoidal modulation:

- `tuning` is a `LaserTuning` (for example `LinearRamp` plus `SinusoidalModulation`), and
  `laser` is an `IntensityModulation`, whose amplitude can be retrieved.
- `demodulator`: `HarmonicExpansion` (quasi-static, fast, the default for retrievals) or
  `DigitalLockIn`, which simulates the lock-in in the time domain with a Butterworth low-pass.
- `observable`: `HarmonicRatio`, `HarmonicOverDC`, `HarmonicProjection`, `Quadratures` or
  `BackgroundSubtracted`.

The lock-in low-pass correlates neighbouring outputs, so a diagonal noise model overstates
the information. `wms_noise_covariance` builds the banded covariance and
`CovariancePrecision` uses it through its sparse Cholesky factor:

```julia
lockin = DigitalLockIn(harmonics = (2,), lowpass = Butterworth(order = 4, cutoff_Hz = 400.0),
                       sampling_rate_Hz = 100e3)
Se = wms_noise_covariance(model.instrument, lockin, detector_variance)
problem = RetrievalProblem(model, y, CovariancePrecision(Se), x_prior, prior_precision)
```

For ratio observables (`HarmonicRatio`, `HarmonicOverDC`, ...), also pass the noise-free
`signals` (a `HarmonicSignals` from `harmonic_signals`), so the covariance can be propagated
to first order. The full script is [examples/wms_retrieval.jl](examples/wms_retrieval.jl).

## Results and diagnostics

A `RetrievalResult` has these fields:

- `x`, `x_prior`: `ComponentVector`s, so `result.x.CH4` works.
- `status` (`Converged`, `MaxIterations`, `Failed`), `message`, `n_iterations`,
  `cost_history`, `rodgers_d2_per_state`.
- `y`, `y_model`, `K` (Jacobian at the solution).
- `posterior_covariance`, `averaging_kernel`, `dfs_per_element` (named like `x`),
  `dfs_total`.
- `chi2_obs_per_dof`, `chi2_total_per_dof`, `prior_influence`, `condition_number`.

The posterior covariance includes the prior term and is evaluated at the solution. Prior
sigmas and posterior sigmas are in state units; divide by the air column for mole fractions.

```julia
save_results("results.jld2", results; times = [m.time for m in measurements])
loaded = load_results("results.jld2")      # plain data, also readable with JLD2 alone
```

## Lookup tables

Line-by-line `HitranVoigt` is exact but slow when pressure or temperature is retrieved. A
`LookupTable` interpolates cross sections precomputed on a (wavenumber, pressure,
temperature) grid. Build one once with vSmartMOM and load it:

```julia
using SpectralFits, JLD2
using vSmartMOM.Absorption

table = read_hitran("hit20_12CH4.par"; mol = 6, iso = 1, ν_min = 6074.0, ν_max = 6080.0)
itp_model = make_interpolation_model(table, Voigt(), 6075.0:0.002:6079.0,
                                     700.0:10.0:1050.0, 250.0:2.0:320.0)
JLD2.@save "ch4_lut.jld2" itp_model

ch4 = load_cross_section_model(LookupTable, "ch4_lut.jld2")
```

Tables written by older Julia or JLD2 versions may not load under Julia 1.12; rebuild them
this way. `read_absco_table(path)` reads OCO ABSCO tables into an `ABSCOTable`. Its H2O
axis is looked up with the dry H2O mole fraction, as the OCO L2 code does.

## Adding a cross-section model

Any type with three methods can be an absorber. Cross sections are in cm2/molecule:

```julia
import SpectralFits: cross_section, depends_on, spectral_range_cm

struct LorentzLine <: AbstractCrossSectionModel
    center_cm::Float64
    strength::Float64
    hwhm_cm_per_hPa::Float64
end
function cross_section(m::LorentzLine, nu, c::GasConditions)
    g = m.hwhm_cm_per_hPa * c.pressure_hPa
    return @. m.strength * (g / pi) / ((nu - m.center_cm)^2 + g^2)
end
depends_on(::LorentzLine) = (:pressure,)          # which conditions it uses
spectral_range_cm(::LorentzLine) = (-Inf, Inf)

absorbers = (CH4 = Absorber(:CH4, LorentzLine(6046.95, 1.6e-20, 5e-5)),)
```

`depends_on` lets the builder precompute the cross section when none of those conditions are
retrieved. New instruments, atmospheres and baselines follow the same pattern; see
[ARCHITECTURE.md](ARCHITECTURE.md).

## Coming from v0.1

| v0.1 | v0.2 |
|---|---|
| `setup` / `inversion_setup` `Dict` | `RetrievalConfig` |
| `get_molecule_info`, `.jld2` LUT loading | `load_cross_section_model(HitranVoigt or LookupTable, path)` |
| `read_DCS_data`, hard-coded 0.9938 | `read_measurements(DCS, path; reader_config)`, `pressure_scale` |
| `calc_DCS_noise` | `ShotNoise()`; `ResidualNoise(...)` |
| `get_measurement`, open interval | `select_window`, closed interval |
| `nonlinear_inversion`, `profile_inversion` | `solve(problem, GaussNewton())` or `LevenbergMarquardt()` |
| `linear = true` (meant log space) | `measurement_space = LogSpace()` |
| `fit_column = true/false` | `state_units = ColumnDensity()` or `VMR()` |
| `"shift"` key | `DCS(calibration = LinearStretch())` |
| `process_all_files` | `Apps.BatchRetrieval.run` |

v0.1 used older physical constants (gravity 9.8196, gas constant 8.314472); v0.2 uses CODATA
2018 and standard gravity, which changes hydrostatic layer columns by about 0.13%.

## Tests, examples and benchmarks

```bash
julia --project=test test/runtests.jl
```

```bash
julia --project=test test/run_one.jl test/core/test_solvers.jl
```

```bash
julia --project=test benchmark/forward_model_benchmarks.jl
```

The full suite takes about 2.5 minutes and uses the HITRAN subset in `test/data`, so it
needs no external line lists. It also runs the scripts in `examples/` (except the batch
example, which needs real data). `test/regression/` checks spectra and retrievals against
the v0.1 code to 1e-10 (see its README).

On a laptop, one DCS forward-model evaluation takes about 0.02 ms and a Jacobian 0.06 ms
with pressure and temperature fixed. Retrieving them costs about 2 ms and 12 ms, because
cross sections are then recomputed line by line.
