# Build, check and write one lookup table.

"""
    LookupTableSummary

What `LookupTables.run` built: `experiment`, `output_file`, `provenance_file`, the grid sizes
`n_wavenumber`, `n_pressure`, `n_temperature`, `file_size_mb`, and the check against the
source at `n_checks` (p, T) points between the nodes: `max_relative_error` (largest
absolute difference over the largest cross section, per point, maximised over the points;
`NaN` for precomputed sources, which have nothing to compare with).
"""
struct LookupTableSummary
    experiment::String
    output_file::String
    provenance_file::String
    n_wavenumber::Int
    n_pressure::Int
    n_temperature::Int
    file_size_mb::Float64
    max_relative_error::Float64
end

_tabulate(source::AbstractCrossSectionModel, c) =
    tabulate_cross_sections(source, c.wavenumber_cm, c.pressure_hPa, c.temperature_K;
                            h2o_dry_mole_fraction = c.h2o_dry_mole_fraction)
_tabulate(source::AbstractArray, c) = source

# relative error of the table against the source at cell-centre (p, T) points
_check(table, source::AbstractArray, c) = NaN
function _check(table, source::AbstractCrossSectionModel, c)
    rng = MersenneTwister(1)
    err = 0.0
    for _ in 1:c.n_checks
        i, j = rand(rng, 1:length(c.pressure_hPa)-1), rand(rng, 1:length(c.temperature_K)-1)
        cond = GasConditions((c.pressure_hPa[i] + c.pressure_hPa[i+1]) / 2,
                             (c.temperature_K[j] + c.temperature_K[j+1]) / 2, c.h2o_dry_mole_fraction)
        exact = cross_section(source, c.wavenumber_cm, cond)
        err = max(err, maximum(abs, cross_section(table, c.wavenumber_cm, cond) .- exact) / maximum(abs, exact))
    end
    return err
end

"""
    LookupTables.run(config::LookupTableConfig) -> LookupTableSummary

Tabulate the source on the grid (threads over the pressure-temperature nodes), fit the cubic
B-spline (`lookup_table_from_grid`), check it against the source between the nodes, and
write `output_file` (`save_lookup_table`; load it with `load_cross_section_model(LookupTable,
path)`) and `<output_file without .jld2>.toml`, the configuration with the build date, git
hash, Julia and SpectralFits versions and the check result.
"""
function run(config::LookupTableConfig)
    @info "Building lookup table" experiment = config.experiment output_file = config.output_file n_nodes = length(config.pressure_hPa) * length(config.temperature_K) threads = Threads.nthreads()
    cs = _tabulate(config.source, config)
    table = lookup_table_from_grid(cs, config.wavenumber_cm, config.pressure_hPa, config.temperature_K;
                                   molecule = config.molecule, isotope = config.isotope,
                                   coefficient_type = config.coefficient_type)
    err = _check(table, config.source, config)
    provenance = merge(config.table, Dict{String,Any}(
        "built" => string(now()), "git_hash" => git_short_hash(), "julia_version" => string(VERSION),
        "spectralfits_version" => string(pkgversion(parentmodule(parentmodule(@__MODULE__)))),
        "max_relative_error" => err))
    mkpath(dirname(abspath(config.output_file)))
    tmp = config.output_file * ".partial"
    save_lookup_table(tmp, table; attributes = provenance)
    mv(tmp, config.output_file; force = config.overwrite)
    prov = string(splitext(config.output_file)[1], ".toml")
    write_config(prov, provenance)
    summary = LookupTableSummary(config.experiment, config.output_file, prov, length(config.wavenumber_cm),
                                 length(config.pressure_hPa), length(config.temperature_K),
                                 filesize(config.output_file) / 1e6, err)
    @info "Wrote lookup table" output_file = config.output_file max_relative_error = err
    return summary
end

"""
    report(summary::LookupTableSummary) -> String
"""
function CoreLayer.report(s::LookupTableSummary)
    return @sprintf("Lookup table %s: %s (%d x %d x %d nodes, %.1f MB), max relative error between nodes %.3g\n",
                    s.experiment, s.output_file, s.n_wavenumber, s.n_pressure, s.n_temperature,
                    s.file_size_mb, s.max_relative_error)
end

export LookupTableSummary
