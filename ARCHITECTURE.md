# Architecture

SpectralFits retrieves trace-gas columns and profiles from spectral measurements by optimal
estimation. One forward model and one set of solvers serve every instrument: dual-comb
spectroscopy (DCS), Fourier transform and grating spectrometers, and open-path
wavelength-modulation spectroscopy (WMS). The design rationale, the map from the v0.1 code,
the decisions on open questions and the implementation notes are in
[docs/architecture_plan.md](docs/architecture_plan.md).

## Layers

Dependencies point inward only. Each layer is a submodule. The layer modules are called
`CoreLayer` and `IOLayer` because `Core` and `IO` would shadow Julia's own names.

- `src/core/` (`SpectralFits.CoreLayer`): pure science. Cross-section models, atmospheres,
  instruments, baselines, the forward model and builder, solvers and diagnostics. No file
  reading or writing, no printing, no global state. `interfaces.jl` holds every abstract type
  and generic function; each other file declares its own exports.
- `src/io/` (`SpectralFits.IOLayer`): every file read and write. The DCS HDF5 reader,
  line-list and lookup-table loaders, the OCO ABSCO reader, results files, run folders
  (`run_folders.jl`), TOML configuration files with `extends` and `[[variants]]`
  (`config_files.jl`), and the parsers that turn configuration tables into core objects,
  shared by every app (`config_specs.jl`: absorbers, noise models, readers, solvers,
  retrieval settings, conditions tables).
- `src/apps/` (`SpectralFits.Apps`): workflows. Each app is a configuration struct, a
  constructor from a configuration table, and `run(config)` returning a summary with its
  diagnostics. `Apps.BatchRetrieval` retrieves every time average of many measurement
  files; `Apps.ProfileOSSE` runs synthetic vertical-profile retrievals; `Apps.LookupTables`
  builds cross-section lookup tables (core `tabulate_cross_sections`,
  `lookup_table_from_grid`; io `subset_line_list`, `save_lookup_table`). `Apps.run_config(path)`
  dispatches a configuration file on its `app` key; `bin/run_config.jl` is the command line.
- `test/` mirrors `src/`, plus `test/integration/` (end-to-end retrievals) and
  `test/regression/` (comparison with the v0.1 reference data).
- `benchmark/forward_model_benchmarks.jl`: forward-model and Jacobian timings.

`using SpectralFits` re-exports the public names of `CoreLayer` and `IOLayer`, and
`ComponentVector`.

## The forward model

A `ForwardModel` is a callable struct with one object per stage, each chosen by type:

1. optical depth from an atmosphere (`HomogeneousPath`, `LayeredColumn`), a named tuple of
   absorbers (cross-section models `HitranVoigt`, `LookupTable`, `ABSCOTable`, wrapped by
   `H2OBroadened` or `Precomputed`), and state units (`ColumnDensity`, `VMR`);
2. transmission;
3. the instrument (`DCS`, `FTS`, `GratingSpectrometer`, `WavelengthModulation`);
4. the measurement space (`LinearSpace`, `LogSpace`);
5. the baseline (`LegendreBaseline`, `NoBaseline`).

`MultiWindowModel` joins several windows that share one state. The state is a
`ComponentVector`. `build_forward_model` does all build-time work: it binds parameter slots
(`Fixed` becomes `FromState` when the prior state has the entry, so retrieving pressure is
only a matter of putting `pressure_hPa` in the state), prepares the instrument and baseline,
chooses and checks the fine grid, precomputes cross sections whose conditions are not
retrieved, and validates the state with errors that name the entry. `RetrievalConfig` plus
`build_problem` assemble a whole problem for one measurement.

## Adding a component

- A new cross-section source: a subtype of `AbstractCrossSectionModel` with `cross_section`,
  `depends_on` and `spectral_range_cm` (see `AnalyticLorentzLine` in `test/helpers.jl`).
- A new atmosphere: a subtype of `AbstractAtmosphere` with `n_layers`, `layer_conditions`,
  `dry_air_column` and `slant_factor`.
- A new instrument: a subtype of `AbstractInstrument` with `prepare`; its prepared form
  implements `apply_instrument`, `wavenumber_coverage_cm` and `n_observations`, and optionally
  `default_fine_grid`, `bind_fine_grid` (operators that depend on the fine grid) and
  `bind_parameters` (parameter slots).
- A new file format: a reader in `src/io/` returning `Measurement`s.

## Estimation

`RetrievalProblem(forward_model, y, obs_precision, x_prior, prior_precision)` validates its
inputs and is solved with `solve(problem, GaussNewton())` or
`solve(problem, LevenbergMarquardt(damping = AdaptiveDamping()))`. Every `RetrievalResult`
carries its status, posterior covariance, averaging kernel, DFS per element (same axes as the
state) and in total, chi squared per degree of freedom, prior influence, condition number and
the Rodgers convergence value; all are computed in one place, `retrieval_result`.
`report(result)` returns a plain-text summary with a CSV table. The posterior covariance is
inverted in Jacobi-scaled form (`scaled_inverse`), since state units differ by up to 1e22.
`correlated_prior_precision` builds vertically correlated priors, and `synthetic_measurement`
the noisy spectra of observing system simulation experiments.

## Configuration files

An experiment is a TOML file, not a script. `read_config` resolves `extends` (a base file,
merged table by table, the extending file wins), relative paths (against the file that sets
them) and `[[variants]]` (one run each, experiment `<experiment>_<name>`; `_replace` and
`_remove` markers replace or drop a table instead of merging it). Each app checks
its keys (unknown keys are errors) and builds every variant before running any.
`run(config)` creates `<output_root>/<yyyy-mm-dd>_<experiment>_<git hash>[_k]/`, copies the
configuration into it, and writes results with `save_results`; `find_run` locates the
newest folder of an experiment. Research repositories keep their configurations and their
analysis scripts; the workflow code lives here.

## Rules

- No file over 500 lines.
- Every exported function has a docstring with argument units and an example.
- Hot paths are type stable: no `Dict` lookups, no abstract fields, names as type parameters.
- Every core function has a test; numerical tests use analytic solutions or pinned results
  with explicit tolerances.

## Running the tests

```
julia --project=test -e 'using Pkg; Pkg.instantiate()'
julia --project=test test/runtests.jl
julia --project=test test/run_one.jl test/core/test_forward_model.jl
```
