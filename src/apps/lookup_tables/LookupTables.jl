"""
    SpectralFits.Apps.LookupTables

Build cross-section lookup tables from a configuration file: a `LookupTableConfig`
(`lookup_table_config`) plus `LookupTables.run(config)`, which tabulates the source
(a HITRAN line list, an ABSCO table, an existing lookup table, or cross sections computed
elsewhere) on the (wavenumber, pressure, temperature) grid, fits the cubic spline, checks it
against the source between the nodes, and writes the table and a provenance file.

```bash
julia -t auto --project=SpectralFit.jl SpectralFit.jl/bin/run_config.jl tables/hit24_ch4win.toml
```
"""
module LookupTables

using Dates
using Printf
using Random
using JLD2
using ...CoreLayer
using ...IOLayer
import ...CoreLayer
import ...IOLayer: config_value, config_table, check_keys

include("lookup_table_config.jl")
include("lookup_table.jl")

end # module LookupTables
