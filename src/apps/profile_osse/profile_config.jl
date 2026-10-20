# Configuration of a synthetic vertical-profile retrieval (observing system simulation
# experiment), read from a configuration table (TOML; see io/config_files.jl).

"""
    ProfileOSSEConfig

One synthetic profile experiment: a layered atmosphere with true gas profiles, a measurement
template (wavenumbers and noise of a real DCS measurement), truth and retrieval cross
sections, a prior with optional vertical correlations, and a solver. Built from a
configuration table by `profile_osse_config`; the table is kept in `table` and copied into
the run folder.
"""
struct ProfileOSSEConfig{A<:NamedTuple, B<:NamedTuple, S<:AbstractSolver, M<:AbstractMeasurementSpace}
    experiment::String
    output_root::String
    table::Dict{String,Any}
    seed::Int
    window_cm::Tuple{Float64,Float64}
    fine_grid_spacing_cm::Float64
    measurement_space::M
    baseline_coefficients::Int
    template::Dict{String,Any}
    noise_source::Symbol            # :template or :snr
    snr::Float64
    atmosphere::NamedTuple
    gases::Vector{Symbol}
    truth_vmr::Dict{Symbol,Vector{Float64}}
    prior_vmr::Dict{Symbol,Float64}
    sigma_vmr::Dict{Symbol,Float64}
    correlation_lengths::Dict{Symbol,Float64}
    correlation_coordinate::Symbol  # :height_km, :pressure_hPa or :none
    nugget::Float64
    baseline_sigma::Float64
    absorbers::A
    truth_absorbers::B
    cross_sections::Dict{String,Any}         # the absorber specs, recorded with the results
    truth_cross_sections::Dict{String,Any}
    solver::S
    target::Symbol
end

const PROFILE_CONFIG_KEYS = ("app", "experiment", "description", "config_file", "variant",
                             "output_root", "seed", "window_cm", "fine_grid_spacing_cm",
                             "measurement_space", "baseline_coefficients", "template", "noise",
                             "atmosphere", "gases", "prior", "solver", "target")

"""
    profile_osse_config(table) -> ProfileOSSEConfig
    profile_osse_config(path::AbstractString; variant = nothing) -> ProfileOSSEConfig

Keys (relative paths are resolved against the file):

```toml
app = "profile_osse"
experiment = "ch4_profile"
output_root = "runs"
seed = 1
window_cm = [6050.0, 6120.0]
fine_grid_spacing_cm = 0.002     # the fine grid is the window at this spacing
measurement_space = "log"        # or "linear"
baseline_coefficients = 5
target = "CH4"                   # gas of the headline diagnostics

[template]                       # wavenumbers and noise of a real measurement
dcs_file = "../data/20160921.h5"
average_index = 10
averaging_minutes = 15
reader = { pressure_scale = 0.9938 }
# noise = { model = "residual", ... }   (noise model of the records; default ShotNoise())

[noise]
source = "template"              # the template's noise; or source = "snr", snr = 2000
                                 # (sigma = mean truth intensity / snr)

[atmosphere]
n_layers = 35
top_m = 10000.0
pressure = { kind = "us_standard" }        # or { kind = "exponential", surface_hPa = 1000.0, scale_height_m = 8500.0 }
temperature = { kind = "lapse_rate", surface_K = 288.15, lapse_K_per_m = 0.0065 }
layer_columns = "ideal_gas"      # layer thickness times number density (v0.1), or "hydrostatic"
mode = "profile"                 # or "isotherm" / "isobar": layer T / p set to the column-weighted mean
# isotherm_temperature_K = 300.0  (isotherm at this temperature instead of the mean)
# column_temperature = { ... }    (temperature profile for the layer columns only)

[gases.CH4]
truth = { kind = "linear", bottom = 2000e-9, top = 1900e-9 }   # or exponential / constant
prior = 1950e-9                  # dry mole fraction, every layer
sigma = 500e-9
correlation_length = 0.5         # in the prior.correlation_coordinate units; omit for none
cross_section = { file = "../spectra/hit08_12CH4.par", molecule = 6, range_cm = [6050.0, 6120.0] }
# truth_cross_section = { ... }   (another line list for the truth; default the same)

[prior]
correlation_coordinate = "height_km"   # or "pressure_hPa"
nugget = 1e-6
baseline_sigma = 1.0

[solver]
name = "levenberg_marquardt"
initial_damping = 40.0
```
"""
function profile_osse_config(d::AbstractDict)
    check_keys(d, PROFILE_CONFIG_KEYS, "top level")
    app = get(d, "app", "profile_osse")
    app == "profile_osse" || throw(ArgumentError("configuration: app is \"$app\", not \"profile_osse\""))
    w = config_value(d, "window_cm")
    window = (Float64(w[1]), Float64(w[2]))
    atmosphere = _atmosphere_from_config(config_table(d, "atmosphere"))
    gases_table = config_table(d, "gases")
    isempty(gases_table) && throw(ArgumentError("configuration [gases]: no gases"))
    gases = Symbol.(sort(collect(keys(gases_table))))
    truth, prior, sigma, lengths = Dict{Symbol,Vector{Float64}}(), Dict{Symbol,Float64}(), Dict{Symbol,Float64}(), Dict{Symbol,Float64}()
    retrieval_specs, truth_specs = Dict{String,Any}(), Dict{String,Any}()
    for g in gases
        t = gases_table[String(g)]
        where = "gases.$g"
        check_keys(t, ("truth", "prior", "sigma", "correlation_length", "cross_section", "truth_cross_section"), where)
        truth[g] = _profile_from_config(config_value(t, "truth"; where = where), atmosphere.height_m, "$where.truth")
        prior[g] = Float64(config_value(t, "prior"; where = where))
        sigma[g] = Float64(config_value(t, "sigma"; where = where))
        haskey(t, "correlation_length") && (lengths[g] = Float64(t["correlation_length"]))
        retrieval_specs[String(g)] = config_value(t, "cross_section"; where = where)
        truth_specs[String(g)] = get(t, "truth_cross_section", t["cross_section"])
    end
    absorbers = absorbers_from_config(retrieval_specs)
    same_truth = all(g -> truth_specs[String(g)] == retrieval_specs[String(g)], gases)
    truth_absorbers = same_truth ? absorbers : absorbers_from_config(truth_specs)
    for (label, set) in (("cross_section", absorbers), ("truth_cross_section", truth_absorbers)), (g, a) in pairs(set)
        lo, hi = spectral_range_cm(a.model)
        (lo <= window[1] && hi >= window[2]) || throw(ArgumentError(
            "configuration [gases.$g.$label]: covers $lo-$hi cm-1, not the window $window (the fine grid is the window)"))
    end
    p = config_table(d, "prior")
    check_keys(p, ("correlation_coordinate", "nugget", "baseline_sigma"), "prior")
    coordinate = Symbol(get(p, "correlation_coordinate", isempty(lengths) ? "none" : "height_km"))
    coordinate in (:height_km, :pressure_hPa, :none) || throw(ArgumentError(
        "configuration [prior]: correlation_coordinate must be \"height_km\", \"pressure_hPa\" or \"none\""))
    noise = config_table(d, "noise")
    check_keys(noise, ("source", "snr"), "noise")
    source = Symbol(get(noise, "source", "template"))
    source in (:template, :snr) || throw(ArgumentError("configuration [noise]: source must be \"template\" or \"snr\""))
    snr = source === :snr ? Float64(config_value(noise, "snr"; where = "noise")) : NaN
    template = config_table(d, "template")
    check_keys(template, ("dcs_file", "average_index", "averaging_minutes", "reader", "noise"), "template")
    config_value(template, "dcs_file"; where = "template")
    target = Symbol(config_value(d, "target"))
    target in gases || throw(ArgumentError("configuration: target $target is not one of the gases $gases"))
    return ProfileOSSEConfig(
        String(config_value(d, "experiment")), String(config_value(d, "output_root")),
        Dict{String,Any}(d), Int(get(d, "seed", 1)), window,
        Float64(config_value(d, "fine_grid_spacing_cm")),
        measurement_space_from_config(String(get(d, "measurement_space", "linear"))),
        Int(get(d, "baseline_coefficients", 5)), Dict{String,Any}(template), source, snr,
        atmosphere, gases, truth, prior, sigma, lengths, coordinate,
        Float64(get(p, "nugget", 1e-6)), Float64(get(p, "baseline_sigma", 1.0)),
        absorbers, truth_absorbers, retrieval_specs, truth_specs,
        solver_from_config(config_table(d, "solver")), target)
end

function profile_osse_config(path::AbstractString; variant = nothing)
    configs = read_config(path; variants = variant === nothing ? nothing : [variant])
    length(configs) == 1 || throw(ArgumentError("$path has $(length(configs)) variants; pick one with variant = name"))
    return profile_osse_config(only(configs))
end

# ---------------------------------------------------------------- atmosphere

function _atmosphere_from_config(t::AbstractDict)
    check_keys(t, ("n_layers", "top_m", "pressure", "temperature", "layer_columns", "mode",
                   "isotherm_temperature_K", "column_temperature"), "atmosphere")
    n = Int(config_value(t, "n_layers"; where = "atmosphere"))
    n >= 2 || throw(ArgumentError("configuration [atmosphere]: n_layers must be at least 2"))
    z = collect(range(0.0, Float64(config_value(t, "top_m"; where = "atmosphere")), length = n))
    T = _temperature_from_config(config_value(t, "temperature"; where = "atmosphere"), z)
    p = _pressure_from_config(config_value(t, "pressure"; where = "atmosphere"), z)
    T_columns = haskey(t, "column_temperature") ? _temperature_from_config(t["column_temperature"], z) : T
    columns = Symbol(get(t, "layer_columns", "ideal_gas"))
    columns in (:ideal_gas, :hydrostatic) || throw(ArgumentError(
        "configuration [atmosphere]: layer_columns must be \"ideal_gas\" or \"hydrostatic\""))
    mode = Symbol(get(t, "mode", "profile"))
    mode in (:profile, :isotherm, :isobar) || throw(ArgumentError(
        "configuration [atmosphere]: mode must be \"profile\", \"isotherm\" or \"isobar\""))
    iso_T = haskey(t, "isotherm_temperature_K") ? Float64(t["isotherm_temperature_K"]) : nothing
    return (height_m = z, pressure_hPa = p, temperature_K = T, column_temperature_K = T_columns,
            layer_columns = columns, mode = mode, isotherm_temperature_K = iso_T)
end

function _temperature_from_config(t::AbstractDict, z)
    kind = String(config_value(t, "kind"; where = "atmosphere.temperature"))
    kind == "lapse_rate" || throw(ArgumentError("configuration [atmosphere.temperature]: kind must be \"lapse_rate\""))
    check_keys(t, ("kind", "surface_K", "lapse_K_per_m"), "atmosphere.temperature")
    return Float64(get(t, "surface_K", 288.15)) .- Float64(get(t, "lapse_K_per_m", 0.0065)) .* z
end

function _pressure_from_config(t::AbstractDict, z)
    kind = String(config_value(t, "kind"; where = "atmosphere.pressure"))
    if kind == "us_standard"          # hydrostatic, constant lapse rate (US Standard 1976 troposphere)
        check_keys(t, ("kind", "surface_hPa", "surface_K", "lapse_K_per_m"), "atmosphere.pressure")
        p0, T0, lapse = Float64(get(t, "surface_hPa", 1013.25)), Float64(get(t, "surface_K", 288.15)),
                        Float64(get(t, "lapse_K_per_m", 0.0065))
        exponent = GRAVITY_M_PER_S2 * MOLAR_MASS_DRY_AIR_KG_PER_MOL / (GAS_CONSTANT_J_PER_MOL_K * lapse)
        return p0 .* ((T0 .- lapse .* z) ./ T0) .^ exponent
    elseif kind == "exponential"
        check_keys(t, ("kind", "surface_hPa", "scale_height_m"), "atmosphere.pressure")
        return Float64(get(t, "surface_hPa", 1000.0)) .* exp.(-z ./ Float64(get(t, "scale_height_m", 8500.0)))
    end
    throw(ArgumentError("configuration [atmosphere.pressure]: kind must be \"us_standard\" or \"exponential\", got \"$kind\""))
end

function _profile_from_config(t::AbstractDict, z, where)
    kind = String(config_value(t, "kind"; where = where))
    if kind == "linear"
        check_keys(t, ("kind", "bottom", "top"), where)
        return collect(range(Float64(t["bottom"]), Float64(t["top"]), length = length(z)))
    elseif kind == "exponential"
        check_keys(t, ("kind", "surface", "scale_height_m"), where)
        return Float64(t["surface"]) .* exp.(-z ./ Float64(t["scale_height_m"]))
    elseif kind == "constant"
        check_keys(t, ("kind", "value"), where)
        return fill(Float64(t["value"]), length(z))
    end
    throw(ArgumentError("configuration [$where]: kind must be \"linear\", \"exponential\" or \"constant\", got \"$kind\""))
end

export ProfileOSSEConfig, profile_osse_config
