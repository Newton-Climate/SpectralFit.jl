# SpectralFits.jl re-architecture plan

Status: proposal, not yet implemented. The current code in [src/](../src) is unchanged.
Audience: anyone working on SpectralFits.jl, including the people rewriting the downstream
scripts in `DCS_experiments` and `OpenPathGHG`. The one-page summary of the target layout is
in [ARCHITECTURE.md](../ARCHITECTURE.md); this document holds the detail and the plan for getting
there.

How to use this document:

- Before opening a pull request for the re-architecture, find its phase in section 5 and
  implement only that phase. Each phase lists the tests it must add.
- When an implementation choice disagrees with this plan, update the plan in the same pull
  request, so the plan stays accurate.
- When a question in section 8 is answered, record the answer here and delete the question.

Contents:

1. Goals and non-goals
2. Problems in the current code that drive the design
3. Target design: types and interface methods
4. Map from the current code to the new code
5. Implementation phases, each a separate pull request with its tests
6. How known bugs are prevented, and which test covers each
7. Type stability and performance
8. Open questions
9. Example: a DCS script under the new API
10. Example: an open-path wavelength-modulation forward model under the new API

Notation. Wavenumber is in cm-1, pressure in hPa, temperature in K, column density in
molecules per cm2, path length in cm, and mole fractions are dry-air mole fractions in
mol/mol unless a name says otherwise. New public names are plain ASCII with units in the
name where there is any doubt (`path_length_cm`, `pressure_hPa`), following the
research-software naming conventions, so they read cleanly with a screen reader. Estimation
names follow the optimal-estimation conventions: `x_prior`, `prior_precision` (the inverse
of Sa), `obs_precision` (the inverse of Se), `K` for the Jacobian, `posterior_covariance`,
`averaging_kernel`.

## 1. Goals and non-goals

Goals:

- One retrieval stack for any spectral instrument: dual-comb spectroscopy (DCS), Fourier
  transform spectrometers (FTS), grating spectrometers, and open-path wavelength-modulation
  spectroscopy (WMS). A new instrument, atmosphere or cross-section source is a new type plus
  a few methods, with no edits to the forward model or the solvers.
- Choices that are now runtime flags become types, resolved by multiple dispatch: measurement
  space (the old `"linear"` flag), state units (`"fit_column"`), spectroscopy special cases
  (`"adjust_ch4_broadening"`, `"use_OCO_table"`), and whether pressure and temperature are
  retrieved.
- Every retrieval result carries its diagnostics: posterior covariance, averaging kernel,
  degrees of freedom for signal (DFS) per element and in total, chi squared per degree of
  freedom, prior influence, and the Rodgers convergence value.
- Core code is pure: no file input or output, no printing, no global state. File formats live
  in `io/`; workflows live in `apps/`.
- No file over 500 lines; tests mirror the source tree.

Non-goals:

- Backward compatibility. There are no deprecated wrappers and no shims for the old
  `Dict`-based API. Downstream scripts are rewritten against the new API (sections 9 and 10).
- New radiative transfer physics. Scattering, solar spectra and surface reflectance stay out
  of scope; the atmosphere types leave room for them.
- GPU support beyond what vSmartMOM already provides for cross-sections.

## 2. Problems in the current code that drive the design

Structure:

- Three forward-model generators that differ only in atmosphere and window handling:
  `generate_forward_model`, `generate_profile_model`, `generate_lhr_model` in
  [src/forward_model.jl](../src/forward_model.jl).
- Three solvers that duplicate covariance set-up, the iteration loop and result assembly:
  `nonlinear_inversion`, `profile_inversion`, `adaptive_inversion` in
  [src/inversion.jl](../src/inversion.jl).
- State vectors are `OrderedDict{String, Union{FT, Vector{FT}}}`, flattened and rebuilt on
  every forward-model call by `assemble_state_vector!`, which also has to know the baseline
  length from the settings. Every lookup is a hash plus a type-unstable `Union`.
- Settings are `Dict{String, Any}` (`RetrievalSetup`), read with `haskey` and `check_param`
  inside the forward model, so misspelled keys are silently ignored.
- Spectroscopy special cases are `if` branches on HITRAN molecule numbers inside
  `calculate_transmission`, with `println` in the hot path.
- Solvers print with `@show` and `println`; the batch driver writes files from inside the
  package.

Correctness issues found while reading the code. The first four are fixed on the unmerged
branch `fix/retrieval-bugs`; this plan does not depend on that branch, and section 6 says how
the new design rules each one out.

1. `profile_inversion` computes the posterior covariance without the prior term, as
   `inv(K' Se_inv K)`.
2. VMR mode (`fit_column=false`) computes the dry-air column `vcd_dry` and never uses it, so
   mole fractions are multiplied directly into cross-sections.
3. Log-space noise weighting divides by the square of the log intensity when the measurement
   holds log intensities, and `adaptive_inversion` overwrites `measurement.intensity` with the
   model.
4. `fit_pressure!`, `fit_temperature!` and `fit_pT!` are dead code that refer to undefined
   names (`OCO_interp`, `CO₂` as a key).
5. `take_time_average!` bins with a closed interval at both ends, so a sample exactly on a bin
   edge is averaged into two bins, and bins with one sample are dropped silently.
6. `adaptive_inversion` never rejects a step that increases the cost, and its predicted-cost
   expression uses the already-updated state, so the gain ratio that sets the damping is
   wrong.
7. `nonlinear_inversion` has no prior but stores `Sₐ⁻¹ = ones(length(grid))` (a vector of
   the wrong length) in the result, and builds the default noise weighting with
   `linear=false` whatever the setting is.
8. H2O is treated inconsistently: column mode computes a dry mole fraction,
   `H2O / (vcd_wet - H2O)`, while VMR mode computes the dry column as `vcd_wet * (1 - q)`,
   which treats `q` as a wet mole fraction.
9. The DCS path length 195017 is labelled "meters" in
   [src/read_data.jl](../src/read_data.jl), but `calc_vcd` treats it as cm (the Loschmidt test
   in [test/test_utils.jl](../test/test_utils.jl) passes 1.0 for 1 cm). A 1.95 km round trip
   is the physical value, so the number is right and the label is wrong. New code names it
   `path_length_cm`.
10. `make_vcd_profile` has a special case when the first and last pressures are equal that
    replaces the temperatures with half-level thicknesses of the temperature profile.
11. `OCO_spectra` builds a four-dimensional grid tuple but scales the interpolant over three
    axes, and `calculate_transmission` calls it with four arguments in a different order, so
    the OCO table path cannot work as written.
12. `run_inversion` builds a forward model and passes it to `fit_spectra`, which ignores it and
    builds another; `fit_spectra` always calls `nonlinear_inversion`, so a batch of profile
    retrievals is impossible.
13. `compute_legendre_poly` asserts at least two terms, so a constant-only baseline cannot be
    used.
14. `DopplerShift` multiplies vectors as matrices and is never called.

## 3. Target design

### 3.1 The pipeline

Every forward model is the same chain of stages, each chosen by type:

1. Optical depth on a fine wavenumber grid: for each layer of the atmosphere and each
   absorber, the slant column of the gas times its cross-section at that layer's conditions,
   summed.
2. Transmission: the exponential of minus the optical depth.
3. Instrument: maps fine-grid transmission to the instrument's observable. For DCS this is
   interpolation onto the measured wavenumbers; for FTS and grating instruments a convolution
   with the instrument line shape; for WMS a time-domain laser simulation followed by lock-in
   demodulation.
4. Measurement space: identity or natural log of the instrument output.
5. Baseline: a multiplicative or additive polynomial in the measurement space, or none.

The stages are generic functions; `ForwardModel` is a parametric callable struct that holds
one object per stage, so the compiler specializes the whole chain for each combination.

### 3.2 Parameter slots: fixed or retrieved, decided at build time

Many quantities are sometimes known and sometimes retrieved: pressure and temperature, a
spectral stretch, WMS laser modulation parameters, baseline coefficients. Today this is a
`haskey(x, "pressure")` test on every call. The new design uses one small mechanism for all
of them:

```julia
abstract type AbstractParameter end
struct Fixed{T} <: AbstractParameter; value::T; end
struct FromState{Name} <: AbstractParameter end   # Name is a Symbol in the state vector

resolve(p::Fixed, x) = p.value
resolve(::FromState{Name}, x) where {Name} = getproperty(x, Name)
```

Structs hold `AbstractParameter` fields through type parameters (for example
`HomogeneousPath{P<:AbstractParameter, T<:AbstractParameter}`). The builder
(`build_forward_model`, section 3.10) looks at the axes of `x_prior` and replaces `Fixed`
with `FromState` where the state contains that name. Because `Name` is a type parameter,
`resolve` compiles to a field access, and there is no runtime branch.

### 3.3 State vector

The state is a `ComponentArrays.ComponentVector`. It behaves as a flat `Vector{Float64}` for
linear algebra and ForwardDiff, and as a named structure for physics code:

```julia
x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1900e-9 * n_air,
                          pressure_hPa = 850.0,
                          baseline = (ch4_window = [1.0, 0, 0, 0, 0],))
x_prior.CH4               # scalar for a homogeneous path
x_prior.baseline.ch4_window
```

Conventions:

- Absorber names are species symbols (`:H2O`, `:CH4`, `:CO2`, `:HDO`). For a layered
  atmosphere each absorber entry is a vector with one value per layer.
- Atmospheric parameters: `:pressure_hPa`, `:temperature_K` (scalar for a homogeneous path;
  vector per layer or a scalar offset for a layered column, see open question 6).
- Instrument parameters sit under `:instrument` (for example `x.instrument.stretch`,
  `x.instrument.i0`).
- Baseline coefficients sit under `:baseline`, one entry per spectral window.
- Prior standard deviations are given as a `ComponentVector` with the same axes, which makes
  a mismatch between prior and state impossible to miss: `prior_precision_from_sigma(sigma)`
  checks the axes and returns `Diagonal(1 ./ sigma .^ 2)`.

This removes `StateVector`, `ProfileStateVector`, `assemble_state_vector!` (all three
methods), `dicttype`, `failed_vector`, `make_shape_params` and `prior_shape_params`.

### 3.4 Cross-section models

```julia
abstract type AbstractCrossSectionModel end

struct GasConditions{T<:Real}
    pressure_hPa::T
    temperature_K::T
    h2o_dry_mole_fraction::T
end

cross_section(model, wavenumber_cm::AbstractVector, conditions::GasConditions) -> AbstractVector
```

Interface for a new cross-section source. It must implement:

- `cross_section(model, wavenumber_cm, conditions)`: returns cm2 per molecule on
  `wavenumber_cm`. It must accept any `Real` element type in `conditions`, so ForwardDiff dual
  numbers pass through when pressure, temperature or H2O are retrieved.
- `depends_on(model)`: a tuple drawn from `(:pressure, :temperature, :h2o)` naming the
  conditions the result depends on. The builder uses it to decide whether the cross-section
  can be precomputed.
- `spectral_range_cm(model)`: `(minimum, maximum)` wavenumber the model is valid for, used to
  check the fine grid at build time.

Provided models:

- `HitranVoigt(model::vSmartMOM.Absorption.HitranModel)`: line-by-line Voigt. Depends on
  pressure and temperature.
- `LookupTable(model::vSmartMOM.Absorption.InterpolationModel)`: interpolation over
  wavenumber, pressure and temperature (the current "Voigt LUT" `.jld2` files).
- `ABSCOTable(interpolant, axes)`: OCO-style absorption coefficient tables over wavenumber, H2O
  broadener mole fraction, temperature and pressure. Depends on all three conditions. This
  replaces the broken `OCO_spectra` path and gets its own tests.
- `H2OBroadened(inner, coefficient)`: evaluates `inner` at an effective pressure of
  `pressure_hPa * (1 + coefficient * h2o_dry_mole_fraction)`, the current
  `adjust_ch4_broadening` rule with coefficient 0.34. Depends on whatever `inner` does plus
  `:h2o`. Any absorber can be wrapped, not only CH4, and there is no molecule-number test.
- `Precomputed(sigma::Matrix)`: cross-sections already evaluated on the fine grid, one column
  per atmospheric layer. Depends on nothing. The builder creates it when none of the
  conditions the inner model depends on are in the state.

`Absorber{M<:AbstractCrossSectionModel}(name::Symbol, model::M)` pairs a state name with a
model. A forward model holds its absorbers as a `NamedTuple`, so a mixed set (one
`Precomputed`, one `H2OBroadened{HitranVoigt}`) stays concretely typed.

Name clash: vSmartMOM exports its own `AbstractCrossSectionModel`. SpectralFits imports
vSmartMOM qualified (`import vSmartMOM.Absorption as Absorption`) and never with `using`, so
the two cannot be confused.

### 3.5 Atmospheres and state units

```julia
abstract type AbstractAtmosphere end
abstract type AbstractStateUnits end
struct ColumnDensity <: AbstractStateUnits end   # state holds molecules per cm2
struct VMR <: AbstractStateUnits end             # state holds dry-air mole fraction, mol/mol
```

Interface for a new atmosphere. It must implement:

- `n_layers(atmosphere)::Int`.
- `layer_conditions(atmosphere, x, layer)::GasConditions`: pressure and temperature resolved
  through parameter slots, and H2O dry mole fraction computed from the state.
- `dry_air_column(atmosphere, x, layer)`: slant dry-air column in molecules per cm2.
- `slant_factor(atmosphere, layer)`: the factor turning a state column into a slant column
  along the light path (1 for an open path, the air mass for a vertical column).

Generic functions written once on top of that interface:

- `gas_column(::ColumnDensity, atmosphere, x, gas, layer) = slant_factor(atmosphere, layer) * x_gas[layer]`
- `gas_column(::VMR, atmosphere, x, gas, layer) = x_gas[layer] * dry_air_column(atmosphere, x, layer)`
- `optical_depth(atmosphere, absorbers, units, x, wavenumber_cm)`: the sum over layers and
  absorbers of `gas_column` times `cross_section`. With every absorber `Precomputed`, this is
  a matrix-vector product per absorber, which is what `layer_optical_depth_operators` does
  today.

Provided atmospheres:

- `HomogeneousPath(; pressure_hPa, temperature_K, path_length_cm)`: one layer, slant factor 1,
  dry-air column from the ideal gas law over the path. For DCS and WMS open paths. A
  convenience method `dry_air_column(atmosphere)` gives the dry-air column of a dry path, for
  writing priors.
- `LayeredColumn(; pressure_hPa::Vector, temperature_K::Vector, solar_zenith_deg)`: layer
  dry-air columns from hydrostatic balance on half-pressure levels, air mass
  `1 / cosd(solar_zenith_deg)`. For profile retrievals.

H2O consistency (issue 8): there is one definition. In column units,
`h2o_dry_mole_fraction = N_H2O / (N_total - N_H2O)`; in VMR units it is the state value. The
dry-air column of a homogeneous path is `N_total / (1 + h2o_dry_mole_fraction)`. A unit test
checks that converting a state from column density to VMR and back is exact, and that both
give the same spectrum.

### 3.6 Instruments

```julia
abstract type AbstractInstrument end
apply_instrument(instrument, wavenumber_fine_cm, transmission, x) -> AbstractVector
```

The instrument owns its observation geometry (which wavenumbers or time samples it observes),
because for WMS the observation axis is time, not wavenumber. The forward model therefore
holds no observation grid of its own.

Interface for a new instrument. It must implement:

- `prepare(instrument, measurement)`: returns a prepared instrument holding everything that
  depends only on the measurement geometry: the observed wavenumbers, interpolation or
  line-shape matrices, lock-in references and filter objects. Called once at build time. This
  replaces the global `Dict` caches `_REF_CACHE` and `_LPF_CACHE` in the OpenPathGHG WMS code.
- `apply_instrument(prepared, wavenumber_fine_cm, transmission, x)`: returns the instrument
  observable, one value per observation. It must be generic in element type for ForwardDiff.
- `wavenumber_coverage_cm(prepared, x_prior)`: `(minimum, maximum)` wavenumber the instrument
  needs, including the line-shape half-width, so the builder can choose or check the fine
  grid.
- `n_observations(prepared)::Int`.

Optional: `default_fine_grid(prepared; oversampling)` suggests a fine grid.

Provided instruments:

- `DCS(; calibration = NoCalibration())`: interpolation of transmission onto the measured
  wavenumbers (cubic B-spline, as now). `LinearStretch()` makes the observed wavenumbers
  `stretch * nu`, with `stretch` a parameter slot; this replaces the `"shift"` key in
  `generate_profile_model`.
- `FTS(; max_path_difference_cm, apodization = Boxcar(), field_of_view_rad = 0.0)`: a sinc
  instrument line shape of width set by the maximum optical path difference, as a banded
  matrix built by `prepare`.
- `GratingSpectrometer(; offsets_cm, response)`: a tabulated line shape, either one for all
  pixels or one per pixel, as a matrix built by `prepare`.
- `WavelengthModulation(...)`: see section 3.7.

For FTS and grating instruments `apply_instrument` is a fixed linear map, a stored sparse
matrix times the transmission. DCS can use the same matrix form when there is no stretch
(open question 4).

### 3.7 Open-path wavelength-modulation spectroscopy

The WMS forward model in `OpenPathGHG/retrieval/instrument.jl` fits this design. It is an
instrument whose observable is a demodulated time series rather than a spectrum. Its stages
become components of one instrument type:

```julia
struct WavelengthModulation{S, L, D, O} <: AbstractInstrument
    sweep::S         # LaserSweep: wavenumber against time
    laser::L         # LaserIntensity: intensity modulation against time
    demodulator::D   # LockIn: harmonics, low-pass cutoff and order
    observable::O    # what is compared with the data
end
```

- `LaserSweep(; start_cm, span_cm, slow_period_s, modulation_amplitude_cm,
  modulation_frequency_Hz, sampling_rate_Hz, branch = RisingBranch())`: produces the
  wavenumber at each sample, a slow triangle plus a fast sine, truncated to one branch. The
  wavenumber trajectory is not monotonic, so instrument interpolation must accept unsorted
  query points; a test covers that. Amplitude, tuning rate and a wavenumber offset can be
  parameter slots.
- `LaserIntensity(; mean_intensity, linear_im, nonlinear_im, linear_phase_rad,
  nonlinear_phase_rad)`: the `I0 * (1 + i0 cos(wt + psi1) + i2 cos(2wt + psi2))` model, every
  coefficient a parameter slot.
- `LockIn(; harmonics = (1, 2), lowpass_cutoff_Hz = 100.0, lowpass_order = 5)`: the
  phase-insensitive digital lock-in (`lockin_xy`), with the reference waveforms and the
  Butterworth filter built once in `prepare`, using the acquisition sampling rate for the
  reference time base.
- Observables, chosen by type and dispatched in `apply_instrument`:
  - `TwoFOverOneF(; background_subtracted = true)`: calibration-free 2f/1f magnitude, with the
    zero-absorption background quadratures computed once in `prepare`.
  - `TwoFOverDC()`: 2f magnitude over the low-passed DC level, which avoids the pole where the
    1f signal nulls.
  - `ProjectedTwoFOverOneF(phase)`: the signed projection a hardware lock-in reports, with
    `phase` a parameter slot.
  - `HardwareLockIn(calibration)`: applies a fitted gain polynomial and offset polynomial from
    `calibrate_hardware_lockin`, which becomes a pure core function returning a
    `HardwareLockInCalibration` struct.
  - `DirectSignal()`: the detector trace itself, for testing.

The same lock-in code builds the observable from a measured detector trace
(`demodulate(prepared, direct_signal)`), so the model and the data pass through identical
causal filters, as the current code requires.

What this changes elsewhere in the design:

- Measurement space and baseline must allow "none": WMS uses `LinearSpace()` (identity) and
  either `NoBaseline()` or an additive polynomial offset in normalized branch position, the
  analogue output drift that `calibrate_hardware_lockin` fits.
- `Measurement` has a generic coordinate axis (section 3.9): sample time in seconds for WMS,
  wavenumber for spectrometers.
- Filtered time series have strongly correlated noise. A diagonal observation covariance on
  every raw sample would overstate the information content and make chi squared meaningless.
  The plan supports decimating the observable to the low-pass bandwidth before the
  retrieval and accepts a full or banded `obs_precision`; open question 9 covers which to use
  by default.
- DSP.jl becomes a dependency, loaded through a package extension so DCS-only users do not
  load it (open question 10). The triangle wave is a two-line function, so Waveforms.jl is
  not needed.
- ForwardDiff through `DSP.filt` with dual-number signals must be checked early (phase 7).

### 3.8 Measurement space and baselines

```julia
abstract type AbstractMeasurementSpace end
struct LinearSpace <: AbstractMeasurementSpace end
struct LogSpace <: AbstractMeasurementSpace end

abstract type AbstractBaseline end
struct NoBaseline <: AbstractBaseline end
struct LegendreBaseline{Mode, Name} <: AbstractBaseline   # Mode: Multiplicative or Additive
    n_coefficients::Int
end
```

Measurement-space methods:

- `to_measurement_space(space, instrument_output)`: identity or `log`.
- `observation_vector(space, measurement)`: identity or `log` of the stored measurement. The
  stored measurement is always in instrument units and never modified.
- `obs_precision(space, measurement; mask)`: the diagonal of the inverse of Se. For
  `LinearSpace` it is `1 / noise_variance`; for `LogSpace` it is
  `intensity^2 / noise_variance`, by first-order error propagation of the log, always using
  the stored linear intensity.

Baseline methods:

- `n_parameters(baseline)`.
- `apply_baseline(baseline, space, y, x)`. Dispatch covers the combinations explicitly:
  multiplicative in linear space is `y .* (P * c)`, multiplicative in log space is
  `y .+ P * c`, additive is `y .+ P * c`. `P` is the Legendre matrix on the observation axis,
  built once in `prepare`.
- `initial_coefficients(baseline, space, measurement)`: the starting guess, the maximum
  intensity for the first coefficient and zeros for the rest, as `prior_shape_params` does
  now.

The baseline is named by its count of coefficients, `n_coefficients`, because the current
`"poly_degree"` is actually the number of Legendre terms. One coefficient (a constant) is
allowed.

### 3.9 Measurements

```julia
struct Measurement{FT<:Real, C<:AbstractVector{FT}, A}
    y::Vector{FT}                # observation in instrument units
    coordinate::C                # wavenumber in cm-1, or sample time in s for WMS
    noise_variance::Vector{FT}   # per observation, instrument units squared
    time::DateTime               # start of the averaging interval
    conditions::A                # NamedTuple, for example (pressure_hPa=..., temperature_K=...)
    metadata::NamedTuple         # n_averaged, source file, machine time
end
```

`Measurement` is immutable and holds no physics settings. Core functions on measurements:

- `select_window(measurement, minimum_cm, maximum_cm)`: a new measurement restricted to an
  open interval, as `get_measurement` does now.
- `time_average(measurements, period::Period)`: bins on half-open intervals, keeps bins with
  one sample, and averages noise variance as the sum over the square of the count. Fixes
  issue 5.
- `estimate_noise_variance(coordinate, intensity; flat_windows, polynomial_order)`: the
  per-record residual estimator from `DCS_experiments/DCS_retrievals/data_driven_noise.jl`,
  replacing `calc_DCS_noise` and its hard-coded window. The old shot-noise scaling model is
  kept as `shot_noise_variance`.

A measurement series is a plain `Vector{Measurement}`; `FrequencyCombDataset` and its matrix
of spectra disappear.

### 3.10 Forward models and the builder

```julia
struct ForwardModel{A<:AbstractAtmosphere, S<:NamedTuple, I, B<:AbstractBaseline,
                    M<:AbstractMeasurementSpace, U<:AbstractStateUnits, G<:AbstractVector}
    atmosphere::A
    absorbers::S
    instrument::I          # prepared instrument
    baseline::B
    space::M
    units::U
    wavenumber_fine_cm::G
end

function (model::ForwardModel)(x::AbstractVector)
    tau = optical_depth(model.atmosphere, model.absorbers, model.units, x, model.wavenumber_fine_cm)
    signal = apply_instrument(model.instrument, model.wavenumber_fine_cm, exp.(-tau), x)
    return apply_baseline(model.baseline, model.space, to_measurement_space(model.space, signal), x)
end

struct MultiWindowModel{W<:Tuple}
    windows::W             # ForwardModels sharing one state, each with its own baseline name
end
(model::MultiWindowModel)(x) = reduce(vcat, map(window -> window(x), model.windows))
```

`build_forward_model(atmosphere, absorbers, instrument, baseline, space, units, measurement,
x_prior; wavenumber_fine_cm = nothing)` does all build-time work:

1. Prepares the instrument for the measurement.
2. Chooses the fine grid from the instrument coverage if none is given, and checks it against
   each cross-section model's range.
3. Replaces `Fixed` parameter slots with `FromState` where `x_prior` has the name.
4. Wraps each absorber's model in `Precomputed` when none of `depends_on(model)` is in the
   state, evaluating it at the atmosphere's layer conditions.
5. Validates the state: every absorber has a state entry and every gas state entry has an
   absorber; layered states have one value per layer; baseline lengths match. Errors name the
   missing or extra entry.

`build_forward_model(config::RetrievalConfig, spectroscopy, measurement, x_prior)` builds a
`ForwardModel` or a `MultiWindowModel` from a configuration (section 3.12).

These three types replace `generate_forward_model`, `generate_profile_model`,
`generate_lhr_model`, `calculate_transmission` (all four methods),
`layer_optical_depth_operators`, `CrossSectionCache` and `cached!`.

### 3.11 Estimation

```julia
struct RetrievalProblem{F, X<:ComponentVector, PO<:AbstractMatrix, PA<:AbstractMatrix}
    forward_model::F
    y::Vector{Float64}           # in measurement space
    obs_precision::PO            # inverse of Se
    x_prior::X
    prior_precision::PA          # inverse of Sa
end

abstract type AbstractSolver end
Base.@kwdef struct GaussNewton <: AbstractSolver
    max_iterations::Int = 30
    rodgers_tolerance::Float64 = 0.01
end
Base.@kwdef struct LevenbergMarquardt{D} <: AbstractSolver
    damping::D = AdaptiveDamping()   # or FixedDamping(gamma)
    scaling::Symbol = :prior         # damping matrix: prior precision, or :marquardt
    max_iterations::Int = 30
    rodgers_tolerance::Float64 = 0.01
end

solve(problem::RetrievalProblem, solver::AbstractSolver) -> RetrievalResult
```

- `RetrievalProblem` validates at construction: dimensions, finite values, positive
  definiteness of both precisions, and a condition number check (rejects above 1e12) of
  `K' Se_inv K + Sa_inv` at the prior.
- Gauss-Newton is the maximum a posteriori form (Rodgers equation 5.9). A pure least-squares
  fit, the current `nonlinear_inversion`, is a problem with a zero prior precision; the
  diagnostics stay defined (the averaging kernel is the identity) and LM falls back to
  Marquardt scaling (open question 7).
- `FixedDamping(gamma)` reproduces `profile_inversion`'s
  `K' Se_inv K + (1 + gamma) Sa_inv`. `AdaptiveDamping(initial, lower_ratio = 0.25,
  upper_ratio = 0.75, factor = 2)` rejects any step that increases the cost, raises the
  damping and retries, and computes the gain ratio from a pure function `predicted_cost` at
  the pre-step state.
- Convergence uses the Rodgers test on accepted steps only: `d2 = dx' * (K' Se_inv K +
  Sa_inv) * dx`, converged when `d2 / n_state < rodgers_tolerance`. Reaching
  `max_iterations` sets the status to `MaxIterations`, never "converged".
- The Jacobian uses `ForwardDiff.jacobian!` with a `JacobianConfig` and result buffer built
  once per problem.
- `solve` neither prints nor catches exceptions. Numerical failures (a singular normal matrix,
  non-finite model output) return a result with status `Failed` and a message. The batch app
  logs and records per-measurement failures.

Result:

```julia
struct RetrievalResult{X<:ComponentVector, PO<:AbstractMatrix, PA<:AbstractMatrix}
    x::X
    x_prior::X
    status::RetrievalStatus          # Converged, MaxIterations, Failed
    message::String
    n_iterations::Int
    rodgers_d2_per_state::Float64    # final value of the convergence statistic
    cost_history::Vector{Float64}
    y::Vector{Float64}
    y_model::Vector{Float64}
    K::Matrix{Float64}
    obs_precision::PO
    prior_precision::PA
    posterior_covariance::Matrix{Float64}
    averaging_kernel::Matrix{Float64}
    dfs_per_element::X               # same axes as x, so dfs.CH4 works
    dfs_total::Float64
    chi2_obs_per_dof::Float64        # observation term over (n_obs - n_state)
    chi2_total_per_dof::Float64      # full cost over (n_obs - n_state), the skill's definition
    prior_influence::Float64         # 1 - dfs_total / n_state
    condition_number::Float64
end
```

All diagnostics are computed by functions in `core/diagnostics.jl`, called by every solver
through one `retrieval_result(problem, x, K, history)` constructor:
`posterior_covariance(K, obs_precision, prior_precision) = inv(Symmetric(K' * obs_precision * K + prior_precision))`,
`averaging_kernel = posterior_covariance * K' * obs_precision * K`, `gain_matrix`,
`dfs_per_element = diag(averaging_kernel)`, `rodgers_d2`, `prior_influence`. No solver
computes a posterior itself, so issue 1 cannot come back in one solver.
`report(result; state_names)` returns the optimal-estimation reporting block as a plain-text
string with a CSV table; printing it is the caller's choice.

### 3.12 Configuration

`RetrievalConfig` is a typed keyword struct that replaces `RetrievalSetup`. It holds
instances of the types above, not strings or flags, and validates in its constructor with
errors that name the field:

```julia
Base.@kwdef struct RetrievalConfig{M<:AbstractMeasurementSpace, U<:AbstractStateUnits,
                                   I<:AbstractInstrument, S<:AbstractSolver}
    windows_cm::Vector{Tuple{Float64,Float64}}
    baseline_coefficients::Vector{Int}           # one per window
    baseline_mode::Symbol = :multiplicative
    measurement_space::M = LinearSpace()
    state_units::U = ColumnDensity()
    instrument::I = DCS()
    retrieve_pressure::Bool = false
    retrieve_temperature::Bool = false
    h2o_broadening::Dict{Symbol,Float64} = Dict()   # absorber => coefficient, build time only
    fine_grid_spacing_cm::Union{Nothing,Float64} = nothing
    masked_windows_cm::Vector{Tuple{Float64,Float64}} = []
    mask_mode::Symbol = :deweight                # or :drop
    baseline_prior::Symbol = :from_measurement   # or :as_given
    solver::S = GaussNewton()
end
```

`build_problem(config, absorbers, measurement, x_prior, prior_sigma)` is the one-call entry
for a single measurement: it builds the forward model, the observation vector and its
precision in the configured measurement space, and the prior precision. Baseline
coefficients in `x_prior` are replaced by `initial_coefficients` from the measurement when
the configuration field `baseline_prior = :from_measurement` (the default), so one prior
serves a whole batch.

The `Dict` field is read only by the builder, never in the forward model. `RetrievalConfig`
lives in core because it describes how core objects are assembled; it contains no paths.
File locations, averaging period and parallelism belong to the app configuration
(section 3.13).

### 3.13 Layout

Dependencies point inward: `apps` use `io` and `core`; `io` uses `core` types; `core` uses
nothing else in the package. Each layer is a submodule so the rule is enforced by the module
system.

- `src/SpectralFits.jl`: module file, includes the submodules, re-exports the public API.
- `src/core/` (module `SpectralFits.Core`):
  - `physical_constants.jl`: `SPEED_OF_LIGHT_CM_PER_S`, `GAS_CONSTANT_J_PER_MOL_K`,
    `AVOGADRO_PER_MOL`, `GRAVITY_M_PER_S2`, `MOLAR_MASS_DRY_AIR_KG_PER_MOL`,
    `MOLAR_MASS_H2O_KG_PER_MOL`. Replaces `c`, `r`, `Nₐ`, `g₀` and the unused index constants.
  - `parameters.jl`: `Fixed`, `FromState`, `resolve`.
  - `air_columns.jl`: number density, homogeneous-path and hydrostatic layer columns,
    half-pressure levels.
  - `spectroscopy/cross_section_models.jl`: the abstract type, `GasConditions`, `HitranVoigt`,
    `LookupTable`, `Absorber`.
  - `spectroscopy/cross_section_wrappers.jl`: `H2OBroadened`, `Precomputed`, `ABSCOTable`.
  - `atmospheres.jl`: `HomogeneousPath`, `LayeredColumn`, state units, `gas_column`,
    `optical_depth`.
  - `instruments/interpolation.jl`: shared interpolation operators.
  - `instruments/dcs.jl`, `instruments/fts.jl`, `instruments/grating.jl`.
  - `instruments/wms_laser.jl`: sweep and intensity models.
  - `instruments/wms_lockin.jl`: lock-in, observables, `demodulate`,
    `calibrate_hardware_lockin`.
  - `baselines.jl`: Legendre matrices, baselines, measurement spaces, `obs_precision`, masks.
  - `measurements.jl`: `Measurement`, `select_window`, `time_average`.
  - `noise_estimation.jl`.
  - `forward_model.jl`: `ForwardModel`, `MultiWindowModel`, the call methods.
  - `forward_model_builder.jl`: `build_forward_model`, validation.
  - `state_vector.jl`: state helpers and `prior_precision_from_sigma`.
  - `retrieval_config.jl`.
  - `retrieval_problem.jl`.
  - `solvers.jl`: `GaussNewton`, `LevenbergMarquardt`, damping types, `solve`.
  - `diagnostics.jl`: posterior, averaging kernel, DFS, chi squared, Rodgers test, `report`.
  - `retrieval_result.jl`.
- `src/io/` (module `SpectralFits.IO`):
  - `dcs_hdf5.jl`: `read_measurements(::Type{DCS}, path; reader_config)`, handling the three
    known field-name layouts, the pressure scale factor and the path length through a
    `DCSReaderConfig`, not constants.
  - `wms_acquisition.jl`: reader for raw WMS detector and hardware lock-in traces.
  - `line_lists.jl`: `load_cross_section_model(HitranVoigt, par_path; molecule, isotope,
    range_cm)` and `load_cross_section_model(LookupTable, jld2_path)`.
  - `absco_tables.jl`: OCO ABSCO HDF5 reader returning `ABSCOTable`.
  - `retrieval_results_jld2.jl`: `save_results` and `load_results`, storing plain arrays and
    named tuples so files do not depend on struct layouts.
- `src/apps/batch_retrieval/` (module `SpectralFits.Apps.BatchRetrieval`):
  - `batch_config.jl`: `BatchRetrievalConfig` (data directory or file list, file pattern,
    reader, averaging period, noise estimator, the `RetrievalConfig`, prior, output
    directory, parallelism).
  - `batch_retrieval.jl`: `run(config)::BatchRetrievalSummary`. Creates the run folder
    `runs/<date>_<experiment>_<git hash>/`, copies the configuration, writes a manifest, logs
    with `@info`, retrieves every measurement with `pmap`, and writes results. The returned
    summary has a `report(summary)` method giving counts of converged, unconverged and
    failed retrievals and the distribution of chi squared and DFS as plain text. Replaces
    `process_all_files`, `run_inversion` and `fit_spectra`.
- `test/` mirrors `src/`: `test/core/...`, `test/io/...`, `test/apps/...`, plus
  `test/regression/`.

## 4. Map from the current code to the new code

[src/SpectralFits.jl](../src/SpectralFits.jl): rewritten. Drops the re-exports of
`OrderedCollections`, `Dates`, `Statistics`, `JLD2` and `vSmartMOM`.

[src/constants.jl](../src/constants.jl):

- `c`, `r`, `Nₐ`, `g₀` become descriptive constants in `core/physical_constants.jl`.
- `H₂O_ind`, `CH₄_ind`, `CO₂_ind`, `HDO_ind`, `temperature_ind`, `pressure_ind`,
  `windspeed_ind` are deleted (state access is by name).

[src/types.jl](../src/types.jl):

- `StateVector`, `ProfileStateVector`, `RetrievalSetup`: deleted; replaced by
  `ComponentVector` and `RetrievalConfig`.
- `MolecularMetaData`, `Molecule`: deleted; replaced by `Absorber` and cross-section models.
- `AbstractResults`, `InversionResults`, `FailedInversion`: replaced by `RetrievalResult` with
  a status.
- `AbstractEnv`, `SurfaceEnv`, `ProfileEnv`: replaced by `HomogeneousPath` and
  `LayeredColumn`.
- `AbstractDataset`, `FrequencyCombDataset`, `AbstractMeasurement`,
  `FrequencyCombMeasurement`: replaced by `Measurement`.
- `SimpleInterpolationModel`: deleted; `LookupTable` wraps vSmartMOM's `InterpolationModel`.

[src/spectroscopy.jl](../src/spectroscopy.jl):

- `get_molecule_info` (both methods), `load_interp_model`: replaced by
  `load_cross_section_model` in `io/line_lists.jl`.
- `setup_molecules`: replaced by a `NamedTuple` of `Absorber`s.
- `calculate_cross_sections`, `calculate_cross_sections!`, `construct_spectra`,
  `construct_spectra!`, `construct_spectra_by_layer`, `construct_spectra_by_layer!`, and
  the globals `p_default`, `T_default`: deleted; `cross_section(model, grid, conditions)` and
  `Precomputed` cover them.

[src/forward_model.jl](../src/forward_model.jl):

- `calculate_transmission` (four methods), `layer_optical_depth_operators`: replaced by
  `optical_depth`.
- `apply_instrument(grid, spectra, grid)`: replaced by `apply_instrument(::PreparedDCS, ...)`.
- `calc_polynomial_term`: replaced by `apply_baseline` with a precomputed Legendre matrix.
- `DopplerShift`, `fit_pressure!`, `fit_temperature!`, `fit_pT!`: deleted.
- `CrossSectionCache`, `cached!`: deleted; `Precomputed` is decided at build time.
- `generate_forward_model`, `generate_profile_model`, `generate_lhr_model`: replaced by
  `build_forward_model`, `ForwardModel` and `MultiWindowModel`.

[src/inversion.jl](../src/inversion.jl):

- `make_obs_error`: replaced by `obs_precision(space, measurement; mask)`.
- `make_prior_error`: replaced by `prior_precision_from_sigma`.
- `failed_vector`, `failed_inversion`: deleted; `RetrievalResult` has a status.
- `nonlinear_inversion`, `profile_inversion`, `adaptive_inversion`: replaced by
  `solve(problem, GaussNewton())` and `solve(problem, LevenbergMarquardt(...))`.
- `fit_spectra`, `run_inversion`, `process_all_files` (both methods): replaced by the batch
  app's `run(config)`.

[src/utils.jl](../src/utils.jl):

- `find_indexes`, `find_measurement_grid`: replaced by `select_window`.
- `vcd_pressure`, `calc_vcd` (both methods), `half_pressure_levels`, `make_vcd_profile`:
  replaced by functions in `core/air_columns.jl`, without the equal-pressure special case.
- `dicttype`, `assemble_state_vector!` (three methods), `make_shape_params`,
  `prior_shape_params`, `check_param`: deleted.
- `compute_legendre_poly`: replaced by `legendre_matrix(n_points, n_coefficients)`, allowing
  one coefficient.
- `calc_gain_matrix`, `calc_χ²`: replaced by `gain_matrix` and fields of `RetrievalResult`.
- `calc_DCS_noise` (both methods): replaced by `estimate_noise_variance` and
  `shot_noise_variance`.
- `find_mask`: kept as an internal helper of `obs_precision`.

[src/read_data.jl](../src/read_data.jl):

- `read_DCS_data`: replaced by `read_measurements(DCS, path; reader_config)` in
  `io/dcs_hdf5.jl`.
- `take_time_average!`: replaced by `time_average` in core (pure, non-mutating).
- `get_measurement` (both methods), `data2measurements`: replaced by `select_window` on a
  `Vector{Measurement}`.
- `OCO_spectra`: replaced by `io/absco_tables.jl` and `ABSCOTable`.

Files deleted outright: [src/example_run.jl](../src/example_run.jl) and
[src/test_grids.jl](../src/test_grids.jl) (they load the old `RadiativeTransfer` package),
the editor backups `src/#constants.jl#`, `src/#read_data.jl#`, `src/#test.jl#`,
`src/#utils.jl#`, `src/.#read_data.jl`, `src/types.jl~`, `src/utils.jl~`, and the stale
top-level scripts `calc_noise.jl`, `example_run.jl`, `forward_model.jl`, `inversion.jl`,
`profile.jl`, `read_data.jl`, `test_files.jl`, `test_profile.jl`. `.gitignore` gains `*~`,
`#*#` and `.#*`.

Dependencies:

- Added: ComponentArrays, DSP (as a weak dependency with an extension).
- Removed: Revise (a development tool, never a package dependency), Parameters, NCDatasets,
  DistributedArrays, RecursiveArrayTools, StatsBase, OrderedCollections and ProgressMeter
  (none are used, or only for progress output the apps do not need). Polynomials is removed
  once the noise estimator uses a small least-squares fit.
- Kept: ForwardDiff, DiffResults, LinearAlgebra, Interpolations, HDF5, JLD2, Dates,
  Statistics, Distributed, vSmartMOM.

## 5. Implementation phases

Work happens on a long-lived branch `rearchitecture`; each phase is one pull request into it,
with its tests passing. The branch merges into `master` after phase 10. Tests use the HITRAN
subset [test/data/hit20_subset.par](../test/data/hit20_subset.par) (H2O, 12CH4 and 12CO2 in
6076 to 6084 cm-1 and 6224 to 6232 cm-1). All tolerances are explicit.

### Phase 0: reference results and clean-up

- Tag the current `master` as `v0.1-legacy` (local tag; open question 1 on pushing it).
- Add `test/regression/generate_reference_v0_1.jl`, run once against the tag, which writes
  `test/regression/reference_v0_1.jld2` with plain arrays: forward-model spectra for the
  CH4 window and the CO2 window, and retrieved states, chi squared and posterior covariances
  from `nonlinear_inversion` (column units, linear space, no prior) and the state and chi
  squared from `profile_inversion`. Only configurations unaffected by issues 1 to 3 are
  recorded; the profile posterior covariance is deliberately excluded.
- Delete the stale files listed in section 4, remove unused dependencies, add the
  `.gitignore` entries.
- Delete the old source and its tests, and add the empty submodule skeleton. From here on the
  package does not provide the old API.

Tests: the package loads; the regression file exists and loads with JLD2 alone.

### Phase 1: constants, air columns, parameter slots, state units

Tests:

- Loschmidt number from the homogeneous-path density at 1013.25 hPa and 273.15 K, relative
  tolerance 1e-3.
- Hydrostatic layer column equals `delta_p / (m_dry * g)` exactly.
- Uniform levels give uniform layer thickness, including the boundaries.
- Column-density and VMR states convert into each other exactly (relative 1e-14), with and
  without H2O.
- `resolve(Fixed(3.0), x) == 3.0`; `resolve(FromState{:a}(), x) == x.a`; both pass
  `@inferred`.

### Phase 2: cross-section models

Tests:

- `HitranVoigt` equals `vSmartMOM.Absorption.absorption_cross_section` on the same grid,
  pressure and temperature to machine precision.
- `Precomputed` built at given conditions equals the inner model at those conditions.
- `H2OBroadened(inner, 0.0)` equals `inner`; with coefficient `c` and mole fraction `q` it
  equals `inner` at pressure `p * (1 + c * q)`.
- ForwardDiff derivatives of `HitranVoigt` with respect to pressure and temperature agree
  with central finite differences (relative 1e-5).
- `LookupTable` built from the subset agrees with `HitranVoigt` inside its grid to the
  interpolation tolerance, set when the test table is made.
- `ABSCOTable` on a synthetic table that is linear in each axis reproduces the linear
  function exactly.
- `depends_on` is correct for every model and wrapper.
- `@inferred cross_section(model, grid, conditions)` for Float64 and dual conditions.

### Phase 3: atmospheres, baselines, measurement spaces, DCS instrument

Tests:

- Beer's law on a single line. A test-only cross-section model `AnalyticLorentzLine` (which
  also shows that a new model needs only the three interface methods) with a known area: for
  a homogeneous path with no instrument effect, the transmission equals
  `exp(-N * sigma(nu))` exactly, and the integrated absorbance equals the column times the
  line strength (relative 1e-6 on a fine grid).
- A one-layer `LayeredColumn` with the same slant column gives the same optical depth as a
  `HomogeneousPath`.
- The layered `Precomputed` path and the direct per-layer loop agree to 1e-12.
- Legendre matrix: orthogonality on a fine grid, and it reproduces the current
  `compute_legendre_poly` values (pinned numbers copied into the test).
- Every baseline and space combination in the dispatch table, including one coefficient.
- `LogSpace` precision: a Monte Carlo of `log(y + noise)` has variance `sigma^2 / y^2` within
  3 standard errors; `obs_precision` never reads a log-transformed value.
- DCS interpolation reproduces a cubic polynomial exactly, and works on unsorted query points.
- `LinearStretch` with `stretch = 1` equals no calibration.

### Phase 4: forward model, builder and multi-window model

Tests:

- ForwardDiff Jacobian against central finite differences for every state element: each gas,
  pressure, temperature, stretch and each baseline coefficient (relative 1e-4).
- With zero gas columns the model returns the baseline; absorption only lowers the signal.
- Column units and VMR units give identical spectra for equivalent states (issue 2).
- With pressure and temperature absent from the state, every absorber is `Precomputed`; with
  them present, none is. Checked with `isa` on the built model.
- A multi-window model with zero absorbers returns each window's own baseline.
- Builder errors: a missing absorber, an extra state entry, a wrong layer count, and a fine
  grid not covering the instrument each raise an error naming the problem.
- `@inferred model(x)` for Float64 and dual states; an allocation budget for the Float64 call
  on the CH4 window.
- Regression: spectra match `reference_v0_1.jld2` to relative 1e-10 for the CH4 and CO2
  windows.

### Phase 5: estimation and diagnostics

Tests on linear-Gaussian problems, where the answer is known in closed form (random `K`,
`Sa`, `Se` with a fixed seed):

- Gauss-Newton returns `x_prior + inv(K' Se_inv K + Sa_inv) K' Se_inv (y - K x_prior)` after
  one step, and the second step has `d2 = 0`, so it stops after two iterations.
- `posterior_covariance` equals `inv(K' Se_inv K + Sa_inv)`; the averaging kernel equals
  `I - posterior_covariance * Sa_inv`; DFS equals its trace (relative 1e-10) (issue 1).
- Levenberg-Marquardt with fixed or adaptive damping reaches the same solution; on a linear
  problem the gain ratio is exactly 1.
- Adaptive damping rejects a step that increases the cost, on a constructed nonlinear problem
  (Rosenbrock-like) where the undamped step overshoots (issue 6).
- Over 500 noise realizations, the mean of `chi2_obs_per_dof` is 1 within 3 standard errors,
  and the true state lies inside the 2-sigma posterior interval at the expected rate.
- A zero prior precision gives the least-squares solution and an identity averaging kernel.
- `RetrievalProblem` rejects mismatched dimensions, a non-positive-definite precision and a
  condition number above 1e12, with messages naming the argument.
- Reaching `max_iterations` reports `MaxIterations`.
- `report` output contains no Greek letters and contains a CSV block with one row per state
  element.

### Phase 6: retrievals on the HITRAN subset

Tests:

- Noise-free column retrieval of H2O, CH4, pressure and temperature from a perturbed prior
  recovers the truth to relative 1e-4 (the current integration test).
- Noisy retrieval: chi squared per degree of freedom between 0.7 and 1.3; retrieved CH4 within
  4 posterior sigma of the truth.
- Inverse-crime guard: truth spectra are generated on a fine grid ten times finer than the
  retrieval grid, so agreement is not guaranteed by construction.
- Masked windows lower the precision of the masked observations by the configured factor in
  both measurement spaces.
- Profile retrieval with fixed and adaptive damping: the total CH4 column within 1 percent of
  the truth, and closer to the truth than the prior.
- Regression against `reference_v0_1.jld2`: column retrievals match states to relative 1e-6
  and chi squared to 1e-6; the no-prior posterior covariance matches to 1e-6. The profile
  state matches within the convergence tolerance; its posterior covariance is checked against
  the corrected formula, not the reference.
- `@inferred solve(problem, GaussNewton())`.

### Phase 7: FTS, grating and WMS instruments

Tests:

- FTS: the sinc line shape integrates to 1 (relative 1e-6); its full width at half maximum is
  `1.207 / (2 * max_path_difference_cm)` (relative 1e-3); convolving a monochromatic line
  returns the line shape.
- Grating: a tabulated Gaussian line shape reproduces the analytic Gaussian convolution of an
  analytic line (relative 1e-6).
- WMS laser: the sweep's wavenumber trajectory matches the triangle-plus-sine formula; a
  pinned comparison with `generate_spectral_grid` and `generate_intensity` from OpenPathGHG
  on one configuration.
- WMS lock-in: a synthetic signal with known 1f and 2f amplitudes and phases is recovered by
  the lock-in within the filter's settling tolerance after the transient; the demodulation of
  measured and simulated traces uses identical filter objects.
- WMS observables: with zero absorption the background-subtracted 2f/1f observable is zero; a
  pinned regression of `TwoFOverDC` against the OpenPathGHG implementation on the HITRAN
  subset.
- ForwardDiff through the WMS chain (including `DSP.filt` with dual numbers) agrees with
  finite differences for CH4 column and the intensity-modulation parameters. If `DSP.filt`
  does not accept dual numbers, the phase implements the filter as a direct-form difference
  equation in core.
- `calibrate_hardware_lockin` recovers a known gain, phase and offset from a synthetic trace.

### Phase 8: input and output

Tests:

- The DCS reader handles all three HDF5 field layouts (written by a fixture builder), converts
  frequency in Hz to cm-1, applies the configured pressure scale, and reads temperature in
  degrees C to K.
- `time_average`: half-open bins; a sample on a bin edge is counted once; single-sample bins
  are kept; noise variance averages as the sum over the count squared (issue 5).
- `estimate_noise_variance` recovers a known white-noise variance (within 5 percent for 10^4
  points).
- Results round-trip through `save_results` and `load_results`, including diagnostics.
- The ABSCO reader returns the table written by a fixture builder.

### Phase 9: batch app

Tests:

- `BatchRetrievalConfig` rejects a missing data directory, empty windows, a negative averaging
  period and a prior whose axes do not match the state, naming the field.
- End to end on synthetic HDF5 files: `run(config)` creates the run folder with the
  configuration copy and manifest, writes one results file per input file, skips files not
  matching the pattern, and records a failed measurement without stopping the batch.
- The same run with two worker processes gives identical results to the serial run.

### Phase 10: documentation and merge

- Update [ARCHITECTURE.md](../ARCHITECTURE.md) to describe the implemented layout and drop its status note, update the
  README, docstrings with units and examples on all exported names.
- Add `benchmark/forward_model_benchmarks.jl` (CH4 window, DCS; profile; WMS) with timings
  recorded in the pull request.
- Port `DCS_experiments/DCS_retrievals/run_merlin.jl` as the worked example (section 9).
- Merge `rearchitecture` into `master` and tag `v0.2.0`.

## 6. How known bugs are prevented

- Issue 1, posterior covariance without the prior. Design: one `posterior_covariance` function
  in `core/diagnostics.jl`, called by the shared result constructor; no solver builds its own.
  Tests: phase 5 closed-form posterior; phase 6 profile posterior against the formula.
- Issue 2, VMR mode ignoring the dry-air column. Design: units are a type; `gas_column`
  dispatches on it, and VMR has no method that returns the state value unscaled. Tests:
  phase 1 unit round trip; phase 4 column versus VMR spectra.
- Issue 3, log-space noise weighting and mutation of the measurement. Design: `Measurement`
  is immutable and always in instrument units; `obs_precision(::LogSpace, measurement)` uses
  the stored linear values; `RetrievalProblem` is immutable, and no solver rebuilds the
  observation covariance. Tests: phase 3 Monte Carlo of log noise; phase 6 masked windows in
  both spaces.
- Issue 4, dead pressure and temperature fitting code. Design: retrieving pressure or
  temperature is only a matter of state membership through parameter slots; there is no
  special code path to rot. Tests: phase 4 Jacobian columns for pressure and temperature;
  phase 6 noise-free recovery of pressure and temperature.
- Issue 5, time-averaging bins. Test: phase 8.
- Issue 6, adaptive damping. Tests: phase 5 gain ratio on a linear problem and step rejection.
- Issue 7, wrong prior stored by the least-squares solver. Design: the result holds the
  problem's actual precisions. Test: phase 5 zero-prior case.
- Issue 8, H2O mole fraction. Test: phase 1 round trip with H2O.
- Issue 9, path length units. Design: the field is `path_length_cm`. Test: phase 8 reader test
  checks the stored value and its use in the dry-air column.
- Issues 10 to 14 are removed with the code that has them; issue 11 is replaced by a tested
  `ABSCOTable` (phase 2).

## 7. Type stability and performance

`@code_warntype` and `@inferred` targets, checked in the tests (and with JET.jl's
`@test_opt` where it works with vSmartMOM):

- `model(x)` for `ForwardModel` and `MultiWindowModel`, with Float64 and dual states.
- `optical_depth`, `cross_section` for each model, `apply_instrument` for each prepared
  instrument, `apply_baseline` for each combination.
- `resolve` for both parameter kinds.
- `solve(problem, solver)` returns a concrete `RetrievalResult`.

Avoiding dynamic lookups in hot paths:

- The state is a `ComponentVector`, whose axes are part of its type, so `x.CH4` is a
  compile-time offset. Names used in hot paths come from type parameters (`FromState{Name}`,
  `LegendreBaseline{Mode, Name}`, `NamedTuple` absorber keys), never runtime `Symbol` values.
- Absorbers are a `NamedTuple` and loops over them are `map` or `sum` over tuples, which the
  compiler unrolls; a `Vector{Absorber}` would have an abstract element type.
- No `Dict`, `haskey`, `Any` field or string comparison runs per forward-model call;
  configuration is turned into types by the builder.

Cross-section caching:

- With pressure and temperature fixed, cross-sections are computed once per measurement by
  the builder (`Precomputed`). This is the dominant saving and replaces the runtime cache.
- In batch runs, measurements whose pressure and temperature fall within a configurable
  tolerance can share one `Precomputed`, or the app can use a `LookupTable` instead of
  line-by-line Voigt.
- When pressure or temperature are retrieved, ForwardDiff evaluates the Voigt profile with dual
  numbers whose width is the chunk size, once per chunk. With 100 baseline coefficients this
  repeats the expensive line-by-line calculation for every chunk although it depends only on
  two state elements. A later optimization: give `cross_section` a dual-number method that
  computes the value and the two partial derivatives once and builds the dual output, or
  split the Jacobian into a ForwardDiff part for gas, pressure and temperature and an analytic
  part for the baseline, which is linear.

Other notes:

- Instrument matrices (FTS, grating, and DCS without stretch) are built once in `prepare` as
  sparse matrices; applying them is a sparse matrix-vector product, which ForwardDiff handles
  directly.
- The Legendre matrix is built once per window.
- Jacobian configuration and result buffers are allocated once per problem; the current
  `adaptive_inversion` allocates them every iteration.
- Forward-model work buffers for Float64 calls can be preallocated; for dual calls, use
  PreallocationTools.jl's `DiffCache` if allocation shows up in profiles.
- WMS: references and filters are built in `prepare`. Traces are about 10^5 samples, so the
  Jacobian is best taken after decimation to the low-pass bandwidth.
- Batch: line lists and models are loaded once per worker, not shipped in closures with every
  task; use `CachingPool` with `pmap`.
- Keep `Float64` throughout; the struct type parameters allow other float types but the tests
  only cover Float64.

## 8. Open questions

1. May the `v0.1-legacy` tag be pushed to GitHub, or should it stay local?
2. Physical constants: keep the current values (gravity 9.8196, gas constant 8.314472,
   Avogadro 6.0221415e23) for regression parity, or switch to CODATA 2018 and standard gravity
   9.80665 after the regression tests pass? Standard gravity shifts hydrostatic columns by
   about 0.1 percent.
3. What is the 0.9938 factor applied to DCS pressure in `read_DCS_data`: a sensor
   calibration specific to one campaign? It becomes a `DCSReaderConfig` field either way;
   the question is its default.
4. DCS interpolation: keep the cubic B-spline (exact regression parity), or switch to a local
   cubic interpolation stored as a sparse matrix, which is faster and identical in form to the
   FTS and grating paths but differs from the current results at about 1e-6?
5. Should H2O be retrieved as a dry-air mole fraction in VMR units (the convention proposed
   here), or as a wet mole fraction?
6. Temperature in a layered column: a full profile (one value per layer) or a scalar offset to
   a prior profile, as TCCON does? Both fit the parameter-slot design; which is the default?
7. Is the no-prior least-squares mode still needed, or should every retrieval carry a prior
   (possibly very weak)?
8. Masked windows: keep de-weighting by a factor of 1e-5 (the current behavior), or drop the
   masked observations, which makes the degrees of freedom in chi squared correct?
9. WMS noise: decimate the observable to the low-pass bandwidth and use a diagonal
   covariance, or keep every sample with a banded covariance from the filter response?
10. Should the WMS code (and DSP.jl) live in SpectralFits as a package extension, as planned,
    or in OpenPathGHG on top of the SpectralFits interface?
11. Parallelism in the batch app: `pmap` over processes as now, threads, or both selectable in
    the configuration?
12. Is the OCO ABSCO path still wanted? If not, `ABSCOTable` and its reader are dropped from
    phases 2 and 8.
13. Spectral windows today use open intervals (a point exactly on a window edge is excluded).
    Keep that?

## 9. Example: a DCS script under the new API

`DCS_experiments/DCS_retrievals/run_merlin.jl` today builds a `RetrievalSetup` dictionary, a
dictionary state, and calls a custom batch loop to override the noise. Under the new API the
noise estimator is part of the app configuration:

```julia
using SpectralFits
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, run
using Dates, Distributed

ch4 = load_cross_section_model(LookupTable, "../spectra/merlin_12CH4.jld2")
h2o = load_cross_section_model(LookupTable, "../spectra/hit20_H2O_narrow.jld2")
absorbers = (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))

retrieval = RetrievalConfig(
    windows_cm = [(6076.05, 6077.95)],
    baseline_coefficients = [5],
    state_units = ColumnDensity(),
    measurement_space = LinearSpace(),
    instrument = DCS(),
    fine_grid_spacing_cm = 0.003,
    solver = GaussNewton(),
)

# prior: pressure and temperature are not in the state, so they come from the measurement
n_air = dry_air_column(HomogeneousPath(pressure_hPa = 850.0, temperature_K = 285.0,
                                       path_length_cm = 195017.0))
x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1900e-9 * n_air,
                          baseline = (window_1 = zeros(5),))   # first coefficient set per measurement
prior_sigma = ComponentVector(H2O = 1.0 * n_air, CH4 = 1e-6 * n_air,
                              baseline = (window_1 = fill(1e3, 5),))

config = BatchRetrievalConfig(
    experiment = "merlin-r6",
    data_files = ["../data/20160921.h5"],
    reader = DCSReaderConfig(path_length_cm = 195017.0),
    noise = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)],
                          polynomial_order = 3),
    averaging_period = Minute(15),
    absorbers = absorbers,
    retrieval = retrieval,
    x_prior = x_prior,
    prior_sigma = prior_sigma,
    output_root = "../data/runs",
)

summary = run(config)
print(report(summary))
```

The custom `process_with_empirical_noise` loop and `data_driven_noise.jl` are no longer
needed. Interactive work on one measurement uses the same pieces directly:

```julia
measurements = read_measurements(DCS, "../data/20160921.h5"; reader_config = config.reader)
m = select_window(time_average(measurements, Minute(15))[1], 6076.05, 6077.95)
problem = build_problem(retrieval, absorbers, m, x_prior, prior_sigma)
result = solve(problem, GaussNewton())
result.x.CH4 / n_air, sqrt(result.posterior_covariance[2, 2]) / n_air, result.dfs_per_element.CH4
```

## 10. Example: an open-path WMS forward model under the new API

The OpenPathGHG simulation in `retrieval/instrument.jl` becomes a configuration of core types:

```julia
wms_measurement = read_measurements(WavelengthModulation, "branch_0001.csv")[1]  # io/wms_acquisition.jl

sweep = LaserSweep(start_cm = 6046.6, span_cm = 0.6, slow_period_s = 0.1,
                   modulation_amplitude_cm = 0.05, modulation_frequency_Hz = 10e3,
                   sampling_rate_Hz = 2.4e6)
laser = LaserIntensity(mean_intensity = 1.0, linear_im = 0.1, nonlinear_im = 0.005,
                       linear_phase_rad = 0.0, nonlinear_phase_rad = 0.0)
instrument = WavelengthModulation(sweep, laser, LockIn(lowpass_cutoff_Hz = 100.0),
                                  TwoFOverDC())

atmosphere = HomogeneousPath(pressure_hPa = 1000.0, temperature_K = 295.0,
                             path_length_cm = 1.0e4)
x_prior = ComponentVector(CH4 = 2.0e-6 * dry_air_column(atmosphere),
                          instrument = (linear_im = 0.1,))   # retrieve one laser parameter

model = build_forward_model(atmosphere, (CH4 = Absorber(:CH4, ch4),), instrument,
                            NoBaseline(), LinearSpace(), ColumnDensity(),
                            wms_measurement, x_prior)
y_model = model(x_prior)
y_observed = demodulate(model.instrument, wms_measurement.y)   # same lock-in as the model
```

The numbers above are placeholders; the OpenPathGHG configuration supplies the real values.
The retrieval then uses the same `RetrievalProblem`, `solve` and diagnostics as DCS.
