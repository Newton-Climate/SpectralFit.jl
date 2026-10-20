# Architecture

Status: this describes the target architecture. The code in `src/` still has the old layout
until the re-architecture lands; the phases, the map from old to new functions, and the open
questions are in [docs/architecture_plan.md](docs/architecture_plan.md).

## What the package does

SpectralFits retrieves trace-gas columns and profiles from spectral measurements by optimal
estimation. One forward model and one set of solvers serve every instrument: dual-comb
spectroscopy, Fourier transform and grating spectrometers, and open-path wavelength-modulation
spectroscopy.

## Layers

Dependencies point inward only. Each layer is a submodule.

- `src/core/` (`SpectralFits.Core`): pure science. Cross-section models, atmospheres,
  instruments, baselines, the forward model, solvers and diagnostics. No file reading or
  writing, no printing, no global state.
- `src/io/` (`SpectralFits.IO`): every file read and write. Instrument data readers, line
  list and lookup table loaders, results files. May use core types.
- `src/apps/` (`SpectralFits.Apps`): workflows. Each app is a configuration struct plus
  `run(config)`. The first app is `batch_retrieval`. Apps use core and io, never each other.
- `test/` mirrors `src/`.

## The forward model

A `ForwardModel` is a callable struct with one object per stage, each chosen by type:

1. optical depth from an atmosphere (`HomogeneousPath`, `LayeredColumn`), a named tuple of
   absorbers (cross-section models such as `HitranVoigt`, `LookupTable`, `ABSCOTable`,
   wrapped by `H2OBroadened` or `Precomputed`), and state units (`ColumnDensity`, `VMR`);
2. transmission;
3. the instrument (`DCS`, `FTS`, `GratingSpectrometer`, `WavelengthModulation`);
4. the measurement space (`LinearSpace`, `LogSpace`);
5. the baseline (`LegendreBaseline`, `NoBaseline`).

`MultiWindowModel` joins several windows that share one state. The state is a
`ComponentVector`; whether a quantity such as pressure is retrieved or fixed is decided once,
when the model is built, from the names in the prior state.

## Adding a component

- A new cross-section source: a subtype of `AbstractCrossSectionModel` with `cross_section`,
  `depends_on` and `spectral_range_cm`.
- A new atmosphere: a subtype of `AbstractAtmosphere` with `n_layers`, `layer_conditions`,
  `dry_air_column` and `slant_factor`.
- A new instrument: a subtype of `AbstractInstrument` with `prepare`, `apply_instrument`,
  `wavenumber_coverage_cm` and `n_observations`.
- A new file format: a reader in `src/io/` returning `Measurement`s.

Signatures and units are in section 3 of the plan.

## Estimation

`RetrievalProblem(forward_model, y, obs_precision, x_prior, prior_precision)` is solved with
`solve(problem, GaussNewton())` or `solve(problem, LevenbergMarquardt(...))`. Every
`RetrievalResult` carries its posterior covariance, averaging kernel, DFS per element and in
total, chi squared per degree of freedom, prior influence, and the Rodgers convergence value.

## Rules

- No file over 500 lines.
- Every exported function has a docstring with argument units and an example.
- Hot paths are type stable: no `Dict` lookups, no abstract fields, names as type parameters.
- Every core function has a test; numerical tests use analytic solutions or pinned results
  with explicit tolerances.
