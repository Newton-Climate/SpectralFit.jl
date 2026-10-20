SpectralFits v0.1-legacy regression reference (Phase 0)
=======================================================

Files
-----
- generate_reference_v0_1.jl : the script that produced the data (old API, tag v0.1-legacy).
- reference_v0_1.jld2         : the data. Only Float64/Int/Bool/String scalars and
                                Float64/String arrays. Loads with `using JLD2` alone
                                (checked with JLD2 0.4.55 and 0.6.7): `d = JLD2.load(file)`
                                gives a Dict with flat keys "group/name".

Generated with Julia 1.12.7, vSmartMOM 1.1.0, Interpolations 0.14.0, ForwardDiff 0.10.39,
JLD2 0.4.55 (the versions in test/Manifest.toml of v0.1-legacy), CPU. Line list:
test/data/hit20_subset.par. Running the script twice gives bit-identical results
(apart from meta/generated_utc).

How it was run: a temporary environment made from the tag's test/Project.toml and
test/Manifest.toml, with the SpectralFits `path = ".."` entry pointed at the legacy checkout;
then `julia --project=<env> generate_reference_v0_1.jl <out.jld2>`. Nothing in the
legacy checkout was modified.


Naming convention for the "linear" flag (IMPORTANT)
---------------------------------------------------
In the old code `setup["linear"] = true` means the model is in LOG space:
    y = log(I_instr(exp(-tau))) + baseline           (additive Legendre baseline)
and `linear = false` means INTENSITY space:
    y = I_instr(exp(-tau)) .* baseline               (multiplicative Legendre baseline)
The keys below use the physical names: "*_intensity" = linear=false, "*_log" = linear=true.
All retrievals recorded are linear=false (intensity space), which the plan calls
"linear space". linear=true retrievals were NOT recorded: nonlinear_inversion then does a
single Gauss-Newton step (max_iter = 1) and the log-space weighting is wrong (plan issue 3).


Groups and keys
---------------
meta/*   strings: description, legacy_tag, julia_version, vSmartMOM_version,
         Interpolations_version, ForwardDiff_version, generated_utc, line_list_file.

const/*  constants used by the old code (src/constants.jl, src/utils.jl):
    c_cm_per_s = 2.99792458e10           (SF.c; not used in these configurations)
    R_J_per_mol_K = 8.314472             (SF.r)
    N_A_per_mol = 6.0221415e23           (SF.N_A)
    g0_m_per_s2 = 9.8196                 (SF.g0)
    dry_air_molar_mass_kg_per_mol = 28.9647e-3, h2o_molar_mass_kg_per_mol = 18.01528e-3
                                          (inside vcd_pressure)

xsec_model/*  cross-section model used for every molecule:
    type = vSmartMOM.Absorption.HitranModel, broadening = Voigt, wing_cutoff_cm-1 = 40,
    vmr = 0.0 (no self broadening), CEF = HumlicekWeidemann32SDErrorFunction,
    architecture = CPU. These are the make_hitran_model defaults that get_molecule_info uses.
    line_selection: lines read with read_hitran(file, mol, iso, nu_min = fine_grid[1],
    nu_max = fine_grid[end]), so ONLY lines centred inside the fine grid contribute
    (lines just outside the grid are dropped even though wing_cutoff is 40 cm-1).
    Cross-sections are absorption_cross_section(model, fine_grid_range, p_hPa, T_K)
    in cm^2/molecule, evaluated on the StepRangeLen fine grid.

ch4_fm/*  and  co2_fm/*  : inputs common to a window
    fine_grid_start/step/stop/length : the model grid, built as start:step:stop
        (CH4 6077.0:0.01:6083.0, CO2 6225.0:0.01:6231.0; 601 points), cm-1.
    fine_grid          : collect() of that range (exact values used).
    meas_grid          : measurement grid, collect(6078.0:0.0067:6082.0) (CH4) or
                         collect(6226.0:0.0067:6230.0) (CO2), 598 points, cm-1.
    pressure_hPa = 850.0, temperature_K = 290.0, pathlength_cm = 195017.0.
    vcd_air_molec_per_cm2 = calc_vcd(p, T, L) = L * p / (R*T) * N_A / 1e4
        (p in hPa, L in cm; = 4.140089023214376e24). Used only to build the gas columns.
    molecules          : ["H2O","CH4"] or ["H2O","CO2"] (this is also the state order).
    mol_<GAS>/mol, /iso: HITRAN molecule and isotope numbers (H2O 1/1, CH4 6/1, CO2 2/1).
    mol_<GAS>/vmr_true : mole fraction used to build the column (H2O 0.01, CH4 1900e-9,
                         CO2 400e-6); column = vmr * vcd_air (no dry-air correction).
    mol_<GAS>/n_lines and mol_<GAS>/lines/<field> : the HitranTable fields actually used
                         (nu_i, S_i, A_i, gamma_air, gamma_self, delta_air, n_air, E'', g', g'',
                         mol, iso), as Float64 vectors.
    xsec_<GAS>_cm2_per_molec : raw cross-section on fine_grid at (p, T) above.

ch4_fm_intensity/*, ch4_fm_log/*, co2_fm_intensity/*, co2_fm_log/* : forward-model runs
    generate_forward_model(x, meas, spectra, setup) with a single homogeneous layer.
    setup_poly_degree = 4 (NUMBER of Legendre coefficients, P0..P3), setup_fit_column = true,
    setup_linear (false/true), setup_use_OCO_table = false, setup_adjust_ch4_broadening = false.
    pT_in_state = false: p and T are not state elements (taken from the measurement).
    state_names        : ["H2O","CH4"|"CO2","shape_parameters[1]",...,"shape_parameters[4]"]
    state_vector       : flat state in that order (gas columns in molecules/cm^2).
    state_<GAS>_molec_per_cm2 : gas columns (same numbers as in state_vector).
    shape_parameters   : intensity: [1.0, 0.02, -0.01, 0.005]; log: [0.0, 0.02, -0.01, 0.005].
    transmission_fine  : exp(-sum_m x_m * sigma_m) on fine_grid.
    transmission_meas  : apply_instrument(fine_grid, transmission_fine, meas_grid).
    baseline           : calc_polynomial_term(4, shape, 598) (see semantics).
    spectrum           : the forward-model output f(x). Exactly equals
                         transmission_meas .* baseline (intensity) or
                         log.(transmission_meas) .+ baseline (log); the script asserts this.
    ch4_fm_intensity/spectrum_pT_in_state_maxabsdiff (and co2): max |difference| when p and T
                         are put in the state at the same values (uncached path) = 0.0.

ch4_inv/*  : nonlinear_inversion on the CH4 window (same grids, p, T, path, cross-sections
             as ch4_fm), column units, intensity space (linear = false), no prior.
    state_names        : ["H2O","CH4","shape_parameters[1..4]"]
    x_true             : [0.01*vcd, 1900e-9*vcd, 1.0, 0.02, -0.01, 0.005]
    x_first_guess      : [0.012*vcd, 1700e-9*vcd, maximum(y_measured), 0, 0, 0]
    noise_sigma = 1e-4, noise_seed_MersenneTwister = 3,
    noise              : 1e-4 .* randn(MersenneTwister(3), 598) (stored, so RNG need not match)
    y_clean            : f(x_true) (equals ch4_fm_intensity/spectrum)
    y_measured         : y_clean + noise (the measurement intensity used)
    setup_poly_degree = 4, setup_fit_column = true, setup_linear = false,
    setup_prior = "none", setup_obs_covariance (string): default Se.
    Se_inv_diag        : diagonal of Se^-1 = 1/sigma^2 = 1e8 everywhere.
    tolerance = 1e-4, max_iter = 30.
    x_retrieved        : returned state (after the last Gauss-Newton update).
    chi2               : returned chi squared (definition below).
    S                  : returned posterior covariance inv(K' Se^-1 K), 6x6.
    K                  : returned Jacobian (598x6), ForwardDiff.
    model              : returned model f (at the last pre-step iterate, see below).
    model_at_x_retrieved : f(x_retrieved), differs from model by 2.5e-13 (converged).
    n_iterations = 4, iterates (6 x 4): the states at which f and K were evaluated, one
                         column per iteration (column 1 = x_first_guess).
    Seed note: seed 42 (the test-helper default) gave a 3.1-sigma CH4 draw with this setup;
    a 15-seed check gave CH4 z-scores within +/-2.2 and Mahalanobis distances 1.8-8.6 for
    6 dof, so seed 3 (z = 0.57) was chosen as a typical draw. chi2 = 1.0338.

prof_inv/*  : profile_inversion on a 3-layer synthetic case, CH4 window.
    n_layers = 3, pressure_hPa = [850, 650, 450], temperature_K = [290, 275, 255]
    vcd_layer_molec_per_cm2 = make_vcd_profile(p, T) (used only to build states; see below)
    sza_deg = 30.0, air_mass_factor = 1/cosd(30) = 1.1547005383792517
    fine_grid*, meas_grid : same as ch4_fm. fine_grid_times_shift_equals_grid = true
                         (the profile model uses 1.0 .* grid; values are identical).
    molecules = ["H2O","CH4"]
    state_names        : H2O[1..3], CH4[1..3], shape_parameters[1..3] (layer columns,
                         molecules/cm^2, layer 1 = 850 hPa).
    x_true             : H2O 0.005*vcd_i, CH4 [1950,1880,1820]e-9*vcd_i, shape [1, 0.01, -0.005]
    x_prior            : prior AND first guess: H2O 0.006*vcd_i, CH4 1800e-9*vcd_i,
                         shape [maximum(y_measured), 0, 0]
    prior_sigma        : H2O 0.005*vcd_i, CH4 300e-9*vcd_i, shape [1, 1, 1]
    Sa_inv_diag        : (1 ./ prior_sigma).^2
    gamma = 0.0        : the old "gamma" regularisation term (set to 0 = plain MAP Gauss-Newton)
    noise_sigma = 1e-4, noise_seed_MersenneTwister = 11, noise, y_clean, y_measured as above.
    setup_poly_degree = 3, setup_fit_column = true, setup_linear = false, default So.
    tolerance = 1e-6, max_iter = 30.
    x_retrieved, chi2 (= 1.0547), model (= f(x_retrieved)), n_iterations = 3,
    iterates (9 x 3)   : states at which K was evaluated each iteration.
    xsec_<GAS>_cm2_per_molec : 601 x 3 raw cross-sections, column i at (p_i, T_i).
    NOT recorded: the posterior covariance (old code omits the prior term, plan issue 1).
    Result check: total CH4 column within 0.13 % of truth (prior was -4.4 %).


Old-code semantics a re-implementation must match
-------------------------------------------------
1. Cross-sections: vSmartMOM absorption_cross_section(HitranModel, fine_range, p_hPa, T_K),
   settings in xsec_model/*. Lines restricted to the fine-grid span.

2. Single-layer optical depth, fit_column = true:
       tau(nu) = sum over molecules m of x_m * sigma_m(nu; p, T)
   with x_m the slant column in molecules/cm^2 (no air-mass factor, no vcd factor).
   calculate_transmission also computes vcd_wet = calc_vcd(p, T, L) and
   vmr_H2O = H2O/(vcd_wet - H2O), but these only feed the CH4-broadening and OCO-table
   branches, both off here. Transmission = exp.(-tau) on the fine grid.
   When p and T are not in the state, cross-sections are cached; putting them in the state
   gives identical numbers.

3. Instrument: apply_instrument(fine_grid, trans, meas_grid) =
       itp  = Interpolations.interpolate(trans, BSpline(Cubic(Line(OnGrid()))))
       sitp = Interpolations.scale(itp, fine_grid)     # fine_grid must be a range
       sitp(meas_grid)
   Cubic B-spline prefiltered interpolation of the TRANSMISSION (not tau, not log) with
   Line(OnGrid()) boundary conditions (Interpolations 0.14.0), evaluated at the measurement
   points. No line-shape convolution.

4. Baseline: calc_polynomial_term(n, c, N):
       x = collect(range(-1, stop=1, length=N))   # N = length(meas_grid); index-based, the
                                                  # wavenumber values themselves are not used
       P = compute_legendre_poly(x, n)            # n x N matrix, rows P0..P_{n-1}
           P0 = 1, P1 = x, P_{l+1} = ((2l+1) x P_l - l P_{l-1})/(l+1)
       baseline = transpose(c' * P)               # sum_k c_k P_{k-1}(x)
   n (setup "poly_degree") is the NUMBER of coefficients and must be >= 2 (assert, issue 13).
   It is applied on the measurement grid AFTER interpolation:
       linear = false : y = sitp(meas_grid) .* baseline
       linear = true  : y = log.(sitp(meas_grid)) .+ baseline   (log after interpolation)

5. Profile model (generate_profile_model, fit_column = true):
       A_m[:, i] = amf * sigma_m(nu; p_i, T_i),  amf = 1/cosd(sza)
       tau = sum_m A_m * x_m      (x_m = vector of layer columns in molecules/cm^2)
   With fit_column = true the vcd factor is 1 (input_is_column). Instrument and baseline
   exactly as above on the grid 1.0 .* fine_grid (identical values).
   State flattening: all layers of the first gas, then all layers of the second gas, ...,
   then shape_parameters (the last sum(poly_degree) entries).

6. Layer air columns (only used here to build x_true/x_prior/prior_sigma):
       make_vcd_profile(p, T): dp = half_pressure_levels(p):
           h[k] = (p[k] + p[k-1])/2 for interior half-levels,
           h[1] = p[1] - (h[2] - p[1]),  h[end] = p[end] + (p[end] - h[end-1]),
           dp = abs.(diff(h))       (here [200, 200, 200] hPa)
       vcd_pressure(dp, T, q=0) = 100 * (1-q) * dp / (M * g0 * 100^2),
           M = (1-q)*28.9647e-3/N_A + q*18.01528e-3/N_A  (kg/molecule); T is unused.
   (The special case p[end] == p[1] in make_vcd_profile, issue 10, is avoided.)

7. nonlinear_inversion (no prior), as run:
       Se^-1 = Diagonal(1/meas.sigma^2) (default branch always uses linear=false weighting)
       x_1 = x_first_guess;  f_0 := zeros (initial buffer)
       for i = 1, 2, ... while i <= 30 and delta > 1e-4:
           f_i, K_i = f(x_i), ForwardDiff.jacobian(f, x_i)
           x_{i+1} = x_i + inv(K_i' Se^-1 K_i) K_i' Se^-1 (y - f_i)
           delta = | ||f_i - y|| - ||f_{i-1} - y|| | / ||f_{i-1} - y||;  delta := 1 if i == 1
       returned x     = x_{last+1} (after the final update)
       returned model = f_last  (at x_last, the state BEFORE the final update)
       chi2 = (y - f_last)' Se^-1 (y - f_last) / (m - n),  m = 598, n = 6
       S    = inv(K_last' Se^-1 K_last),  K = K_last
   Cost minimised: (y - f)' Se^-1 (y - f). The stored Sa^-1 in the result is a dummy
   (ones(m), issue 7) and was not recorded.

8. profile_inversion, as run (gamma = 0):
       So^-1 = Diagonal(1/sigma^2) (linear = false), Sa^-1 = Diagonal(1 ./ prior_sigma.^2),
       x_a = x_1 = x_prior
       for i = 1, 2, ... while i <= 30 and delta > 1e-6:
           f_i, K_i = f(x_i), jacobian at x_i
           lhs = K_i' So^-1 K_i + Sa^-1 + gamma*Sa^-1
           rhs = K_i' So^-1 (y - f_i) - Sa^-1 (x_i - x_a)
           x_{i+1} = x_i + lhs \ rhs
           f_{i+1} = f(x_{i+1})
           delta = | ||f_{i+1} - y|| - ||f_i - y|| | / ||f_i - y||;  delta := 1 if i == 1
       returned x = x_final, model = f(x_final),
       chi2 = (y - f(x_final))' So^-1 (y - f(x_final)) / (m - n),  m = 598, n = 9
       (no prior term in chi2). Note chi2 here is evaluated at the returned state, unlike
       nonlinear_inversion, where it is evaluated one step earlier.
   Cost minimised (gamma = 0): (y - f)' So^-1 (y - f) + (x - x_a)' Sa^-1 (x - x_a).

9. Units: wavenumber cm-1; pressure hPa; temperature K; path length cm; columns
   molecules/cm^2; cross-sections cm^2/molecule; intensities dimensionless.


Configurations deliberately not recorded
----------------------------------------
- fit_column = false (VMR units): vcd_dry is computed and not used (issue 2).
- linear = true retrievals: one GN step only and wrong log-space weighting (issue 3).
- adaptive_inversion: wrong gain ratio and no step rejection (issue 6).
- Profile posterior covariance: missing prior term (issue 1).
- CH4 H2O-broadening and OCO-table branches (issue 11 for OCO).
No configuration errored; nothing had to be substituted.
