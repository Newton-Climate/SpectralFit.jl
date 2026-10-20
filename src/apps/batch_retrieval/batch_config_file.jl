# BatchRetrievalConfig from a configuration table (TOML; see io/config_files.jl).

const BATCH_CONFIG_KEYS = ("app", "experiment", "description", "config_file", "variant",
                           "data_directory", "data_files", "file_pattern", "output_root",
                           "averaging_minutes", "parallel", "n_workers", "reader", "noise",
                           "conditions", "absorbers", "retrieval", "solver", "prior")

"""
    batch_config(table) -> BatchRetrievalConfig
    batch_config(path::AbstractString; variant = nothing) -> BatchRetrievalConfig

Batch retrieval configuration from a configuration table (one entry of `read_config`) or a
file with at most one run. Keys (relative paths are resolved against the file):

```toml
app = "batch_retrieval"
experiment = "hit08_ch4"
data_directory = "../data"            # or data_files = ["../data/20160921.h5"] (takes precedence)
file_pattern = '\\.h5\$'                # regular expression on the file name
output_root = "../data/runs"
averaging_minutes = 15
parallel = "processes"                # or "serial"
n_workers = 4

[reader]                              # DCSReaderConfig
path_length_cm = 195017.0
pressure_scale = 0.9938

[noise]                               # optional; replaces the reader's ShotNoise()
model = "residual"
flat_windows_cm = [[6076.10, 6076.60], [6077.55, 6077.95]]

[conditions]                          # optional; e.g. tower p and T, see read_conditions_csv
file = "../data/tower_pT.csv"
max_offset_s = 60

[absorbers.CH4]                       # see absorber_from_config
file = "../spectra/hit08_12CH4.jld2"

[retrieval]                           # see retrieval_config_from_config
window_cm = [6050.0, 6115.0]
baseline_coefficients = 100
retrieve_pressure = true
retrieve_temperature = true

[solver]                              # see solver_from_config
name = "gauss_newton"

[prior]                               # gas values and sigmas are dry mole fractions of the
                                      # reference air column; gases without an absorber are ignored
reference_pressure_hPa = 850.0        # (state units are columns)
reference_temperature_K = 285.0
baseline_sigma = 1e3
pressure_hPa = { value = 850.0, sigma = 100.0 }
temperature_K = { value = 285.0, sigma = 30.0 }
[prior.gases]
CH4 = { value = 1900e-9, sigma = 1e-6 }
```
"""
function batch_config(d::AbstractDict)
    check_keys(d, BATCH_CONFIG_KEYS, "top level")
    app = get(d, "app", "batch_retrieval")
    app == "batch_retrieval" || throw(ArgumentError("configuration: app is \"$app\", not \"batch_retrieval\""))
    noise = haskey(d, "noise") ? noise_model_from_config(d["noise"]) : nothing
    reader = reader_from_config(config_table(d, "reader"))
    conditions = nothing
    if haskey(d, "conditions")
        c = d["conditions"]
        check_keys(c, ("file", "max_offset_s"), "conditions")
        conditions = read_conditions_csv(String(config_value(c, "file"; where = "conditions"));
                                         max_offset = Millisecond(round(Int, 1000 * get(c, "max_offset_s", 60))))
    end
    absorbers = absorbers_from_config(config_table(d, "absorbers"))
    retrieval = retrieval_config_from_config(config_table(d, "retrieval");
                                             solver = solver_from_config(config_table(d, "solver")))
    x_prior, prior_sigma = _prior_from_config(config_table(d, "prior"), absorbers, retrieval, reader)
    files = get(d, "data_files", nothing)
    return BatchRetrievalConfig(
        experiment = String(config_value(d, "experiment")),
        data_files = files === nothing ? nothing : String.(files),
        # data_files wins, so a file can narrow a base file's data_directory
        data_directory = (files === nothing && haskey(d, "data_directory")) ? String(d["data_directory"]) : nothing,
        file_pattern = Regex(String(get(d, "file_pattern", "\\.h5\$"))),
        reader = reader, noise = noise,
        averaging_period = Millisecond(round(Int, 60_000 * config_value(d, "averaging_minutes"))),
        conditions = conditions, absorbers = absorbers, retrieval = retrieval,
        x_prior = x_prior, prior_sigma = prior_sigma,
        output_root = String(config_value(d, "output_root")),
        parallel = Symbol(get(d, "parallel", "processes")),
        n_workers = Int(get(d, "n_workers", 2)))
end

function batch_config(path::AbstractString; variant = nothing)
    configs = read_config(path; variants = variant === nothing ? nothing : [variant])
    length(configs) == 1 || throw(ArgumentError(
        "$path has $(length(configs)) variants; pick one with variant = name"))
    return batch_config(only(configs))
end

function _prior_from_config(t::AbstractDict, absorbers::NamedTuple, retrieval::RetrievalConfig,
                            reader::DCSReaderConfig)
    check_keys(t, ("reference_pressure_hPa", "reference_temperature_K", "baseline_sigma",
                   "pressure_hPa", "temperature_K", "gases"), "prior")
    n_air = homogeneous_air_column(Float64(get(t, "reference_pressure_hPa", 850.0)),
                                   Float64(get(t, "reference_temperature_K", 285.0)),
                                   reader.path_length_cm)
    gases = config_table(t, "gases"; where = "prior")
    values, sigmas = Pair{Symbol,Float64}[], Pair{Symbol,Float64}[]
    for name in keys(absorbers)
        g = config_value(gases, String(name); where = "prior.gases")
        push!(values, name => Float64(config_value(g, "value"; where = "prior.gases.$name")) * n_air)
        push!(sigmas, name => Float64(config_value(g, "sigma"; where = "prior.gases.$name")) * n_air)
    end
    # gases without an absorber are ignored, so a campaign base file can list all of them
    for (flag, key) in ((retrieval.retrieve_pressure, "pressure_hPa"), (retrieval.retrieve_temperature, "temperature_K"))
        flag || continue
        e = config_value(t, key; where = "prior")
        push!(values, Symbol(key) => Float64(config_value(e, "value"; where = "prior.$key")))
        push!(sigmas, Symbol(key) => Float64(config_value(e, "sigma"; where = "prior.$key")))
    end
    baseline_sigma = Float64(get(t, "baseline_sigma", 1e3))
    windows = Tuple(Symbol("window_", i) for (i, n) in enumerate(retrieval.baseline_coefficients) if n > 0)
    counts = [n for n in retrieval.baseline_coefficients if n > 0]
    base_x = NamedTuple{windows}(Tuple(zeros(n) for n in counts))
    base_s = NamedTuple{windows}(Tuple(fill(baseline_sigma, n) for n in counts))
    x = isempty(windows) ? ComponentVector(; values...) : ComponentVector(; values..., baseline = base_x)
    s = isempty(windows) ? ComponentVector(; sigmas...) : ComponentVector(; sigmas..., baseline = base_s)
    return x, s
end

export batch_config
