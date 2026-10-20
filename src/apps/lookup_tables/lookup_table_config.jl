# Configuration of one lookup table (TOML; see io/config_files.jl).

"""
    LookupTableConfig

One lookup table to build: the `source` cross-section model (or a precomputed array), the
grid, the output file and its options. Built by `lookup_table_config`; `table` is the
configuration table, written next to the output.
"""
struct LookupTableConfig{S}
    experiment::String
    table::Dict{String,Any}
    output_file::String
    overwrite::Bool
    wavenumber_cm::StepRangeLen{Float64}
    pressure_hPa::StepRangeLen{Float64}
    temperature_K::StepRangeLen{Float64}
    source::S                       # AbstractCrossSectionModel, or a 3-d array on the grid
    h2o_dry_mole_fraction::Float64
    molecule::Int
    isotope::Int
    coefficient_type::Type
    n_checks::Int
end

const LOOKUP_TABLE_KEYS = ("app", "experiment", "description", "config_file", "variant", "output_file",
                           "overwrite", "wavenumber_cm", "pressure_hPa", "temperature_K", "source",
                           "coefficient_type", "n_checks")

"""
    lookup_table_config(table) -> LookupTableConfig
    lookup_table_config(path::AbstractString; variant = nothing) -> LookupTableConfig

Keys (relative paths are resolved against the file):

```toml
app = "lookup_table"
experiment = "tccon_HDO_ch4win"
output_file = "../tccon_HDO_ch4win.jld2"
overwrite = false                           # refuse to replace an existing file
wavenumber_cm = { start = 6049.0, stop = 6116.0, step = 0.003 }   # start:step:stop, so the
pressure_hPa  = { start = 600.0, stop = 1050.0, step = 10.0 }      # last node may be below stop
temperature_K = { start = 230.0, stop = 350.0, step = 10.0 }
coefficient_type = "Float64"                # or "Float32" (half the size)
n_checks = 4                                # off-node (p, T) points checked against the source

[source]                                    # a HITRAN line list (see absorber_from_config)
file = "../tccon_2020.par"
molecule = 1
isotope = 4
line_margin_cm = 40.0                       # lines within 40 cm-1 of the grid (default)
wing_cutoff_cm = 40
line_shape = "voigt"
relabel = ["491", " 14"]                    # optional: rewrite the 3-character molecule-isotope field
h2o_dry_mole_fraction = 0.0                 # H2O at which the source is evaluated
```

Other sources: an ABSCO table (`file = "OCO_spectra.hdf"`, `gas`, `h2o_indices`), an
existing lookup table (`file = "x.jld2"`), or cross sections computed elsewhere on exactly
this grid (`file = "x_cs.jld2"`, `array_key = "cs"`, plus `molecule`, `isotope`), for
example by a vSmartMOM branch with line shapes the release does not have.
"""
function lookup_table_config(d::AbstractDict)
    check_keys(d, LOOKUP_TABLE_KEYS, "top level")
    app = get(d, "app", "lookup_table")
    app == "lookup_table" || throw(ArgumentError("configuration: app is \"$app\", not \"lookup_table\""))
    grid(key) = _grid_from_config(config_value(d, key), key)
    nu, p, T = grid("wavenumber_cm"), grid("pressure_hPa"), grid("temperature_K")
    src = Dict{String,Any}(config_table(d, "source"))
    isempty(src) && throw(ArgumentError("configuration: missing table [source]"))
    q = Float64(pop!(src, "h2o_dry_mole_fraction", 0.0))
    source, molecule, isotope = _source_from_config(src, nu, p, T)
    out = String(config_value(d, "output_file"))
    endswith(out, ".jld2") || throw(ArgumentError("configuration: output_file must end in .jld2, got $out"))
    ct = String(get(d, "coefficient_type", "Float64"))
    ct in ("Float64", "Float32") || throw(ArgumentError("configuration: coefficient_type must be \"Float64\" or \"Float32\""))
    overwrite = Bool(get(d, "overwrite", false))
    overwrite || !ispath(out) || throw(ArgumentError(
        "configuration: output_file $out exists; set overwrite = true to replace it"))
    return LookupTableConfig(String(config_value(d, "experiment")), Dict{String,Any}(d), out, overwrite,
                             nu, p, T, source, q, molecule, isotope,
                             ct == "Float32" ? Float32 : Float64, Int(get(d, "n_checks", 4)))
end

function lookup_table_config(path::AbstractString; variant = nothing)
    configs = read_config(path; variants = variant === nothing ? nothing : [variant])
    length(configs) == 1 || throw(ArgumentError("$path has $(length(configs)) variants; pick one with variant = name"))
    return lookup_table_config(only(configs))
end

function _grid_from_config(t, key)
    t isa AbstractDict || throw(ArgumentError("configuration: $key must be a table { start, stop, step }"))
    check_keys(t, ("start", "stop", "step"), key)
    a, b, h = Float64(config_value(t, "start"; where = key)), Float64(config_value(t, "stop"; where = key)),
              Float64(config_value(t, "step"; where = key))
    (h > 0 && b > a) || throw(ArgumentError("configuration [$key]: need start < stop and step > 0"))
    r = a:h:b                      # Julia range semantics: the last node is at or below stop
    length(r) >= 4 || throw(ArgumentError("configuration [$key]: need at least 4 nodes for a cubic spline"))
    return r
end

function _source_from_config(src::Dict{String,Any}, nu, p, T)
    file = String(config_value(src, "file"; where = "source"))
    molecule, isotope = Int(get(src, "molecule", -1)), Int(get(src, "isotope", -1))
    if haskey(src, "array_key")                        # cross sections computed elsewhere
        check_keys(src, ("file", "array_key", "molecule", "isotope"), "source")
        isfile(file) || throw(ArgumentError("configuration [source]: cross-section array file not found: $file"))
        cs = JLD2.jldopen(f -> f[String(src["array_key"])], file, "r")
        size(cs) == (length(nu), length(p), length(T)) || throw(DimensionMismatch(
            "configuration [source]: $file holds an array of size $(size(cs)), the grid is $((length(nu), length(p), length(T)))"))
        return Array{Float64,3}(cs), molecule, isotope
    end
    spec = Dict{String,Any}(src)
    relabel = pop!(spec, "relabel", nothing)
    if lowercase(splitext(file)[2]) in (".par", ".txt", ".data")
        margin = Float64(pop!(spec, "line_margin_cm", 40.0))
        spec["range_cm"] = [first(nu), last(nu)]
        spec["line_margin_cm"] = margin
        if relabel !== nothing
            length(relabel) == 2 || throw(ArgumentError("configuration [source]: relabel must be [from, to]"))
            subset = tempname() * ".par"
            subset_line_list(file, subset; range_cm = (first(nu) - margin, last(nu) + margin),
                             relabel = String(relabel[1]) => String(relabel[2]))
            spec["file"] = subset
        end
        isotope = Int(get(spec, "isotope", 1))
        molecule = Int(config_value(spec, "molecule"; where = "source"))
    elseif relabel !== nothing
        throw(ArgumentError("configuration [source]: relabel applies only to HITRAN line lists"))
    end
    return absorber_from_config(:source, spec).model, molecule, isotope
end

export LookupTableConfig, lookup_table_config
