# Synthetic vertical-profile retrieval: build the layered atmosphere, simulate a noisy DCS
# spectrum on a real measurement's wavenumbers, retrieve, and write the run folder.

"""
    ProfileOSSESummary

What `ProfileOSSE.run` did: `experiment`, `run_folder`, `results_file`, `status`,
`n_iterations`, `chi2_obs_per_dof`, `noise_sigma` (intensity units), `snr` (mean intensity
over noise sigma), `target`, and per gas (`Dict{Symbol,Float64}`): `dfs` (degrees of freedom
of the profile), `column_error_percent` (retrieved over true column, minus 1) and
`column_sigma_percent` (posterior sigma of the column). `profile_rmse_percent` is the RMS of
the target's layer mole-fraction errors relative to the truth. `report(summary)` gives text.
"""
struct ProfileOSSESummary
    experiment::String
    run_folder::String
    results_file::String
    status::RetrievalStatus
    n_iterations::Int
    chi2_obs_per_dof::Float64
    noise_sigma::Float64
    snr::Float64
    target::Symbol
    dfs::Dict{Symbol,Float64}
    column_error_percent::Dict{Symbol,Float64}
    column_sigma_percent::Dict{Symbol,Float64}
    profile_rmse_percent::Float64
end

# Template measurements by (file, reader settings, averaging): reading a DCS file takes
# seconds, and a configuration with many variants uses the same one.
const TEMPLATE_CACHE = Dict{Any,Vector{Measurement}}()

"""
    template_measurement(config::ProfileOSSEConfig) -> Measurement

The configured average of the template DCS file, restricted to the window (closed interval).
The template's noise model (`template.noise`, default `ShotNoise()`) gives its noise variance.
"""
function template_measurement(config::ProfileOSSEConfig)
    t = config.template
    noise = haskey(t, "noise") ? noise_model_from_config(t["noise"]) : ShotNoise()
    reader = reader_from_config(config_table(t, "reader"; where = "template"); noise_model = noise)
    minutes = Float64(get(t, "averaging_minutes", 15))
    key = (String(t["dcs_file"]), reader, minutes)
    averages = get!(TEMPLATE_CACHE, key) do
        records = read_measurements(DCS, key[1]; reader_config = reader)
        time_average(records, Millisecond(round(Int, 60_000 * minutes)))
    end
    i = Int(get(t, "average_index", 1))
    1 <= i <= length(averages) || throw(ArgumentError(
        "configuration [template]: average_index $i, the file has $(length(averages)) averages"))
    return select_window(averages[i], config.window_cm...)
end

"""
    profile_atmosphere(config) -> NamedTuple

Layer pressures and temperatures after the mode collapse, layer dry-air columns
(molecules/cm2) from the uncollapsed profile, and heights.
"""
function profile_atmosphere(config::ProfileOSSEConfig)
    a = config.atmosphere
    z, p, T = a.height_m, a.pressure_hPa, a.temperature_K
    h2o = get(config.truth_vmr, :H2O, zeros(length(z)))
    vcd = if a.layer_columns === :ideal_gas
        dz_cm = 100 * (z[2] - z[1])
        homogeneous_air_column.(p, a.column_temperature_K, dz_cm) .* (1 .- h2o)
    else
        hydrostatic_dry_air_column.(layer_pressure_thickness_hPa(p))
    end
    w = vcd ./ sum(vcd)
    n = length(z)
    p_layers = a.mode === :isobar ? fill(sum(w .* p), n) : p
    T_layers = a.mode === :isotherm ?
        fill(something(a.isotherm_temperature_K, sum(w .* T)), n) : T
    return (height_m = z, pressure_hPa = p_layers, temperature_K = T_layers,
            dry_air_column_cm2 = vcd, profile_pressure_hPa = p)
end

function _state(config, values::Dict{Symbol,<:AbstractVector}, baseline)
    gases = NamedTuple{Tuple(config.gases)}(Tuple(values[g] for g in config.gases))
    return ComponentVector(; gases..., baseline = (window_1 = baseline,))
end

"""
    ProfileOSSE.run(config::ProfileOSSEConfig) -> ProfileOSSESummary

1. Create the run folder `<output_root>/<yyyy-mm-dd>_<experiment>_<git hash>/` and copy the
   configuration into it (`configuration.toml`).
2. Build the layered atmosphere and the truth state (layer columns of the true mole
   fractions), simulate the spectrum with the truth cross sections on the template's
   wavenumbers, and add Gaussian intensity noise (`synthetic_measurement`, seed `seed`).
3. Retrieve with the retrieval cross sections from the prior (constant mole fractions), with
   vertically correlated prior covariance when correlation lengths are given
   (`correlated_prior_precision`).
4. Write `results/<experiment>_results.jld2` (`save_results`; the truth, prior sigma, layer
   pressures, temperatures, columns and heights, and the retrieval and truth cross-section
   specs (`cross_sections`, `truth_cross_sections`: gas => file, molecule, ...) in the
   attributes) and `summary.txt`.
"""
function run(config::ProfileOSSEConfig)
    folder = create_run_folder(config.output_root, config.experiment)
    write_config(joinpath(folder, "configuration.toml"), config.table)
    template = template_measurement(config)
    atm = profile_atmosphere(config)
    vcd = atm.dry_air_column_cm2
    space = config.measurement_space
    lo, hi = config.window_cm
    grid = range(lo, hi; step = config.fine_grid_spacing_cm)

    baseline = initial_coefficients(LegendreBaseline(config.baseline_coefficients), space, template)
    n = length(vcd)
    x_true = _state(config, Dict(g => config.truth_vmr[g] .* vcd for g in config.gases), baseline)
    x_prior = _state(config, Dict(g => config.prior_vmr[g] .* vcd for g in config.gases), baseline)
    sigma = _state(config, Dict(g => config.sigma_vmr[g] .* vcd for g in config.gases),
                   fill(config.baseline_sigma, config.baseline_coefficients))

    model(absorbers, x) = build_forward_model(
        LayeredColumn(pressure_hPa = atm.pressure_hPa, temperature_K = atm.temperature_K,
                      dry_air_column_cm2 = vcd),
        absorbers, DCS(), LegendreBaseline(config.baseline_coefficients), space, ColumnDensity(),
        template, x; wavenumber_fine_cm = grid)
    f_truth = model(config.truth_absorbers, x_true)
    f = config.truth_absorbers === config.absorbers ? f_truth : model(config.absorbers, x_prior)

    intensity = space isa LogSpace ? exp.(f_truth(x_true)) : f_truth(x_true)
    noise_sigma = config.noise_source === :template ? sqrt(template.noise_variance[1]) :
                  mean(intensity) / config.snr
    measurement = synthetic_measurement(f_truth, x_true, template, space, noise_sigma;
                                        rng = MersenneTwister(config.seed))

    coordinate = config.correlation_coordinate === :pressure_hPa ? atm.profile_pressure_hPa :
                 atm.height_m ./ 1000
    lengths = config.correlation_coordinate === :none ? Dict{Symbol,Float64}() : config.correlation_lengths
    prior_precision = isempty(lengths) ? Matrix(prior_precision_from_sigma(sigma, x_prior)) :
        correlated_prior_precision(sigma, coordinate, NamedTuple{Tuple(keys(lengths))}(Tuple(values(lengths)));
                                   nugget = config.nugget)
    problem = RetrievalProblem(f, observation_vector(space, measurement), obs_precision(space, measurement),
                               x_prior, prior_precision)
    result = solve(problem, config.solver)

    dfs, col_err, col_sigma = Dict{Symbol,Float64}(), Dict{Symbol,Float64}(), Dict{Symbol,Float64}()
    for g in config.gases
        i = state_indices(result.x, g)
        dfs[g] = sum(result.dfs_per_element[i])
        truth_column = sum(getproperty(x_true, g))
        col_err[g] = 100 * (sum(getproperty(result.x, g)) / truth_column - 1)
        col_sigma[g] = 100 * sqrt(sum(result.posterior_covariance[i, i])) / truth_column
    end
    t = config.target
    rel = getproperty(result.x, t) ./ getproperty(x_true, t) .- 1
    rmse = 100 * sqrt(mean(abs2, rel))

    results_file = joinpath(folder, "results", "$(config.experiment)_results.jld2")
    save_results(results_file, [result]; coordinates = [collect(Float64, measurement.coordinate)],
                 attributes = (experiment = config.experiment, target = String(t),
                               x_true = collect(x_true), x_true_labels = state_labels(x_true),
                               prior_sigma = collect(sigma), prior_sigma_labels = state_labels(sigma),
                               height_m = atm.height_m, pressure_hPa = atm.pressure_hPa,
                               temperature_K = atm.temperature_K, dry_air_column_cm2 = vcd,
                               profile_pressure_hPa = atm.profile_pressure_hPa,
                               mode = String(config.atmosphere.mode), noise_sigma = noise_sigma,
                               snr = mean(intensity) / noise_sigma, seed = config.seed,
                               cross_sections = config.cross_sections,
                               truth_cross_sections = config.truth_cross_sections))
    summary = ProfileOSSESummary(config.experiment, folder, results_file, result.status,
                                 result.n_iterations, result.chi2_obs_per_dof, noise_sigma,
                                 mean(intensity) / noise_sigma, t, dfs, col_err, col_sigma, rmse)
    write(joinpath(folder, "summary.txt"), report(summary) * "\n" * report(result))
    @info "Profile OSSE finished" experiment = config.experiment run_folder = folder status = result.status dfs_target = dfs[t] column_error_percent = col_err[t]
    return summary
end

"""
    report(summary::ProfileOSSESummary) -> String

Plain-text summary with one CSV row per gas: `gas,dfs,column_error_percent,column_sigma_percent`.
"""
function CoreLayer.report(s::ProfileOSSESummary)
    io = IOBuffer()
    println(io, "Profile OSSE report: ", s.experiment)
    println(io, "run_folder: ", s.run_folder)
    @printf(io, "status: %s  iterations: %d  chi2_obs_per_dof: %.4g  noise_sigma: %.4g  snr: %.0f\n",
            s.status, s.n_iterations, s.chi2_obs_per_dof, s.noise_sigma, s.snr)
    @printf(io, "target: %s  profile_rmse_percent: %.4g\n", s.target, s.profile_rmse_percent)
    println(io, "```csv")
    println(io, "gas,dfs,column_error_percent,column_sigma_percent")
    for g in sort(collect(keys(s.dfs)))
        @printf(io, "%s,%.4f,%.5f,%.5f\n", g, s.dfs[g], s.column_error_percent[g], s.column_sigma_percent[g])
    end
    println(io, "```")
    return String(take!(io))
end

export ProfileOSSESummary, template_measurement, profile_atmosphere
