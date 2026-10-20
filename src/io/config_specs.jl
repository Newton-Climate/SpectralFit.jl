# Core objects from configuration tables (see io/config_files.jl). Shared by the apps, so
# every app spells absorbers, noise models, readers and solvers the same way.

"""
    absorber_from_config(name::Symbol, spec) -> Absorber

One absorber from a table with a `file`:

- `.jld2`: a vSmartMOM lookup table (`LookupTable`); optional `key` (default `"itp_model"`).
- `.par` (or `.txt`, `.data`): a HITRAN line list, line-by-line Voigt (`HitranVoigt`); needs
  `molecule` and `range_cm = [lo, hi]` (cm-1); optional `isotope` (default 1, -1 for all),
  `line_margin_cm`, `wing_cutoff_cm`, `line_shape` (`"voigt"` (default), `"lorentz"`,
  `"doppler"`, or `"hartmann_tran"`, which needs a vSmartMOM with `HartmannTran`).
- `.h5`, `.hdf`, `.hdf5`: an OCO ABSCO table (`ABSCOTable`); optional `gas` (HITRAN gas
  number), `range_cm`, `h2o_indices` (1-based, for example `[1]` for the dry-air slice).

```toml
[absorbers.CH4]
file = "../spectra/hit08_12CH4.par"
molecule = 6
range_cm = [6048.0, 6117.0]
```
"""
function absorber_from_config(name::Symbol, spec::AbstractDict)
    where = "absorbers.$name"
    file = String(config_value(spec, "file"; where = where))
    ext = lowercase(splitext(file)[2])
    model = if ext == ".jld2"
        check_keys(spec, ("file", "key"), where)
        load_cross_section_model(LookupTable, file; key = String(get(spec, "key", "itp_model")))
    elseif ext in (".par", ".txt", ".data")
        check_keys(spec, ("file", "molecule", "isotope", "range_cm", "line_margin_cm", "wing_cutoff_cm", "line_shape"), where)
        r = config_value(spec, "range_cm"; where = where)
        load_cross_section_model(HitranVoigt, file; molecule = Int(config_value(spec, "molecule"; where = where)),
                                 isotope = Int(get(spec, "isotope", 1)), range_cm = (Float64(r[1]), Float64(r[2])),
                                 line_margin_cm = Float64(get(spec, "line_margin_cm", 0.0)),
                                 wing_cutoff_cm = Int(get(spec, "wing_cutoff_cm", 40)),
                                 broadening = line_shape_from_config(String(get(spec, "line_shape", "voigt")), where))
    elseif ext in (".h5", ".hdf", ".hdf5")
        check_keys(spec, ("file", "gas", "range_cm", "h2o_indices"), where)
        r = get(spec, "range_cm", nothing)
        h = get(spec, "h2o_indices", nothing)
        read_absco_table(file; gas = haskey(spec, "gas") ? Int(spec["gas"]) : nothing,
                         range_cm = r === nothing ? nothing : (Float64(r[1]), Float64(r[2])),
                         h2o_indices = h === nothing ? nothing : Int.(h))
    else
        throw(ArgumentError("configuration [$where]: file \"$file\" is not a .jld2 lookup table, " *
                            "a .par line list or an .h5/.hdf ABSCO table"))
    end
    return Absorber(name, model)
end

"""vSmartMOM broadening function from a line-shape name (see `absorber_from_config`)."""
function line_shape_from_config(name::AbstractString, where::AbstractString = "")
    name == "voigt" && return Absorption.Voigt()
    name == "lorentz" && return Absorption.Lorentz()
    name == "doppler" && return Absorption.Doppler()
    if name == "hartmann_tran"
        isdefined(Absorption, :HartmannTran) || throw(ArgumentError(
            "configuration [$where]: line_shape \"hartmann_tran\" needs a vSmartMOM with HartmannTran " *
            "(the feature/line-mixing-ht branch); the loaded vSmartMOM has none"))
        return getfield(Absorption, :HartmannTran)()
    end
    throw(ArgumentError("configuration [$where]: line_shape must be \"voigt\", \"lorentz\", \"doppler\" or \"hartmann_tran\", got \"$name\""))
end

"""
    absorbers_from_config(table) -> NamedTuple

Absorbers from a table of absorber tables (`[absorbers.H2O]`, `[absorbers.CH4]`, ...),
in alphabetical order of their names (TOML tables are unordered).
"""
function absorbers_from_config(table::AbstractDict)
    isempty(table) && throw(ArgumentError("configuration [absorbers]: no absorbers"))
    names = Tuple(Symbol.(sort(collect(keys(table)))))
    return NamedTuple{names}(Tuple(absorber_from_config(n, table[String(n)]) for n in names))
end

"""
    noise_model_from_config(table) -> AbstractNoiseModel

`model = "shot"` (optional `flat_window_cm`, `polynomial_order`), `"residual"`
(`flat_windows_cm`, optional `polynomial_order`) or `"constant"` (`variance`).
"""
function noise_model_from_config(t::AbstractDict)
    model = String(config_value(t, "model"; where = "noise"))
    if model == "shot"
        check_keys(t, ("model", "flat_window_cm", "polynomial_order"), "noise")
        w = get(t, "flat_window_cm", [6255.1, 6255.4])
        return ShotNoise(flat_window_cm = (Float64(w[1]), Float64(w[2])),
                         polynomial_order = Int(get(t, "polynomial_order", 3)))
    elseif model == "residual"
        check_keys(t, ("model", "flat_windows_cm", "polynomial_order"), "noise")
        return ResidualNoise(flat_windows_cm = windows_from_config(config_value(t, "flat_windows_cm"; where = "noise")),
                             polynomial_order = Int(get(t, "polynomial_order", 3)))
    elseif model == "constant"
        check_keys(t, ("model", "variance"), "noise")
        return ConstantNoise(Float64(config_value(t, "variance"; where = "noise")))
    end
    throw(ArgumentError("configuration [noise]: model must be \"shot\", \"residual\" or \"constant\", got \"$model\""))
end

"""
    reader_from_config(table; noise_model = ShotNoise()) -> DCSReaderConfig

Keys: `path_length_cm`, `pressure_scale`, `invalid_records` (`"skip"` or `"error"`),
`time_epoch` (ISO 8601, default the LabVIEW epoch).
"""
function reader_from_config(t::AbstractDict; noise_model::AbstractNoiseModel = ShotNoise())
    check_keys(t, ("path_length_cm", "pressure_scale", "invalid_records", "time_epoch"), "reader")
    return DCSReaderConfig(path_length_cm = Float64(get(t, "path_length_cm", 195017.0)),
                           pressure_scale = Float64(get(t, "pressure_scale", 1.0)),
                           noise_model = noise_model,
                           invalid_records = Symbol(get(t, "invalid_records", "skip")),
                           time_epoch = haskey(t, "time_epoch") ? DateTime(String(t["time_epoch"])) : LABVIEW_EPOCH)
end

"""
    solver_from_config(table) -> AbstractSolver

`name = "gauss_newton"` or `"levenberg_marquardt"`; optional `max_iterations`,
`rodgers_tolerance`, and for Levenberg-Marquardt `initial_damping` (adaptive damping) or
`fixed_damping`, and `scaling` (`"prior"` or `"marquardt"`).
"""
function solver_from_config(t::AbstractDict)
    check_keys(t, ("name", "max_iterations", "rodgers_tolerance", "initial_damping", "fixed_damping", "scaling"), "solver")
    name = String(get(t, "name", "gauss_newton"))
    common = (max_iterations = Int(get(t, "max_iterations", 30)),
              rodgers_tolerance = Float64(get(t, "rodgers_tolerance", 0.01)))
    name == "gauss_newton" && return GaussNewton(; common...)
    name == "levenberg_marquardt" || throw(ArgumentError(
        "configuration [solver]: name must be \"gauss_newton\" or \"levenberg_marquardt\", got \"$name\""))
    damping = haskey(t, "fixed_damping") ? FixedDamping(Float64(t["fixed_damping"])) :
              AdaptiveDamping(initial = Float64(get(t, "initial_damping", 1.0)))
    return LevenbergMarquardt(; damping = damping, scaling = Symbol(get(t, "scaling", "prior")), common...)
end

"""`"linear"` -> `LinearSpace()`, `"log"` -> `LogSpace()`."""
function measurement_space_from_config(s::AbstractString)
    s == "linear" && return LinearSpace()
    s == "log" && return LogSpace()
    throw(ArgumentError("configuration: measurement_space must be \"linear\" or \"log\", got \"$s\""))
end

"""
    retrieval_config_from_config(table; solver) -> RetrievalConfig

Keys: `window_cm = [lo, hi]` or `windows_cm = [[lo, hi], ...]` (joined in one retrieval),
`baseline_coefficients` (one number, or one per window), `baseline_mode`,
`measurement_space`, `retrieve_pressure`, `retrieve_temperature`, `h2o_broadening`
(table absorber = coefficient), `fine_grid_spacing_cm`, `fine_grid_margin_cm`,
`masked_windows_cm`, `mask_mode`, `mask_factor`, `baseline_prior`.
"""
function retrieval_config_from_config(t::AbstractDict; solver::AbstractSolver)
    allowed = ("window_cm", "windows_cm", "baseline_coefficients", "baseline_mode", "measurement_space",
               "retrieve_pressure", "retrieve_temperature", "h2o_broadening", "fine_grid_spacing_cm",
               "fine_grid_margin_cm", "masked_windows_cm", "mask_mode", "mask_factor", "baseline_prior")
    check_keys(t, allowed, "retrieval")
    haskey(t, "window_cm") == haskey(t, "windows_cm") && throw(ArgumentError(
        "configuration [retrieval]: give exactly one of window_cm and windows_cm"))
    windows = haskey(t, "window_cm") ? windows_from_config([t["window_cm"]]) : windows_from_config(t["windows_cm"])
    nb = config_value(t, "baseline_coefficients"; where = "retrieval")
    coefficients = nb isa AbstractVector ? Int.(nb) : fill(Int(nb), length(windows))
    spacing = get(t, "fine_grid_spacing_cm", nothing)
    margin = get(t, "fine_grid_margin_cm", nothing)
    return RetrievalConfig(windows_cm = windows, baseline_coefficients = coefficients,
                           baseline_mode = Symbol(get(t, "baseline_mode", "multiplicative")),
                           measurement_space = measurement_space_from_config(String(get(t, "measurement_space", "linear"))),
                           retrieve_pressure = Bool(get(t, "retrieve_pressure", false)),
                           retrieve_temperature = Bool(get(t, "retrieve_temperature", false)),
                           h2o_broadening = Dict{Symbol,Float64}(Symbol(k) => Float64(v) for (k, v) in get(t, "h2o_broadening", Dict())),
                           fine_grid_spacing_cm = spacing === nothing ? nothing : Float64(spacing),
                           fine_grid_margin_cm = margin === nothing ? nothing : Float64(margin),
                           masked_windows_cm = windows_from_config(get(t, "masked_windows_cm", [])),
                           mask_mode = Symbol(get(t, "mask_mode", "deweight")),
                           mask_factor = Float64(get(t, "mask_factor", 1e-5)),
                           baseline_prior = Symbol(get(t, "baseline_prior", "from_measurement")),
                           solver = solver)
end

# ---------------------------------------------------------------- conditions time series

"""
    read_conditions_csv(path; max_offset = Second(60)) -> ConditionsTable

Measurement conditions from a CSV file with a header line: a `time` column (ISO 8601, for
example `2016-09-21T09:14:19`) and one numeric column per condition, named as the
measurement conditions (`pressure_hPa`, `temperature_K`, ...). Lines starting with `#` are
comments.

```
time,pressure_hPa,temperature_K
2016-09-21T09:14:19,821.82,295.32
```
"""
function read_conditions_csv(path::AbstractString; max_offset::Dates.Period = Second(60))
    isfile(path) || throw(ArgumentError("conditions file not found: $path"))
    lines = [l for l in readlines(path) if !isempty(strip(l)) && !startswith(strip(l), "#")]
    length(lines) >= 2 || throw(ArgumentError("$path: needs a header and at least one row"))
    header = strip.(split(lines[1], ','))
    it = findfirst(==("time"), header)
    it === nothing && throw(ArgumentError("$path: no \"time\" column in header $header"))
    columns = [j for j in eachindex(header) if j != it]
    times = DateTime[]
    values = [Float64[] for _ in columns]
    for (n, line) in enumerate(lines[2:end])
        fields = strip.(split(line, ','))
        length(fields) == length(header) || throw(ArgumentError(
            "$path line $(n + 1): $(length(fields)) fields, the header has $(length(header))"))
        push!(times, DateTime(fields[it]))
        for (k, j) in enumerate(columns)
            push!(values[k], parse(Float64, fields[j]))
        end
    end
    nt = NamedTuple{Tuple(Symbol.(header[columns]))}(Tuple(values))
    return ConditionsTable(times, nt; max_offset = max_offset)
end

export absorber_from_config, absorbers_from_config, noise_model_from_config, reader_from_config
export solver_from_config, measurement_space_from_config, retrieval_config_from_config
export read_conditions_csv
