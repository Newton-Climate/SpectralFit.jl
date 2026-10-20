# Reader for OCO ABSCO absorption coefficient tables (HDF5). Replaces v0.1 `OCO_spectra`,
# which could not work (issue 11 in docs/architecture_plan.md).
#
# Layout as HDF5.jl reads it (Julia order, the reverse of the C order in the file):
#   Wavenumber           (n_nu,)                  cm-1
#   Broadener_01_VMR     (n_h2o,)                 H2O mole fraction of wet air, mol/mol
#   Temperature          (n_T, n_p)               K, one temperature grid per pressure level
#   Pressure             (n_p,)                   Pa
#   Gas_XX_Absorption    (n_nu, n_h2o, n_T, n_p)  cm2/molecule

"""
    read_absco_table(path; gas = nothing, range_cm = nothing) -> ABSCOTable

Read an OCO ABSCO HDF5 table into an `ABSCOTable`. `gas` selects the absorption dataset by
its two-digit HITRAN gas number (`gas = 2` reads `Gas_02_Absorption`); by default the file
must hold exactly one `Gas_XX_Absorption` dataset. `range_cm = (minimum, maximum)`, in cm-1,
reads only the wavenumbers inside that interval plus one neighbour on each side, which keeps
memory small for the multi-gigabyte production tables. Pressure is converted from Pa to hPa.
A file without `Broadener_01_VMR` is read as a table with one H2O value, 0. A one-dimensional
`Temperature` is shared by all pressure levels.

```julia
co2 = read_absco_table("co2_v51.h5"; gas = 2, range_cm = (6200.0, 6260.0))
cross_section(co2, 6210.0:0.01:6250.0, GasConditions(850.0, 290.0, 0.01))
```
"""
function read_absco_table(path::AbstractString; gas::Union{Nothing,Integer} = nothing,
                          range_cm::Union{Nothing,Tuple{Real,Real}} = nothing)
    isfile(path) || throw(ArgumentError("ABSCO file not found: $path"))
    return HDF5.h5open(path, "r") do file
        for name in ("Wavenumber", "Temperature", "Pressure")
            haskey(file, name) || throw(ArgumentError(
                "ABSCO file $path has no dataset \"$name\"; datasets: $(keys(file))"))
        end
        gas_name = absco_gas_dataset(file, gas, path)
        nu_all = Vector{Float64}(read(file["Wavenumber"]))
        nu_index = absco_wavenumber_indices(nu_all, range_cm, path)
        pressure_hPa = 0.01 .* vec(Float64.(read(file["Pressure"])))
        temperature_K = Float64.(read(file["Temperature"]))
        h2o = haskey(file, "Broadener_01_VMR") ?
              vec(Float64.(read(file["Broadener_01_VMR"]))) : [0.0]
        dataset = file[gas_name]
        dims = size(dataset)
        sigma = if length(dims) == 4
            Float64.(dataset[nu_index, :, :, :])
        elseif length(dims) == 3   # no broadener axis
            reshape(Float64.(dataset[nu_index, :, :]), length(nu_index), 1, dims[2], dims[3])
        else
            throw(ArgumentError("ABSCO dataset $gas_name in $path has $(length(dims)) " *
                                "dimensions; expected 4 (nu, h2o, T, p) or 3 (nu, T, p)"))
        end
        size(sigma, 1) == length(nu_index) || throw(DimensionMismatch(
            "ABSCO dataset $gas_name in $path does not match the Wavenumber length"))
        table_axes = (wavenumber_cm = nu_all[nu_index], h2o_mole_fraction = h2o,
                      temperature_K = temperature_K, pressure_hPa = pressure_hPa)
        ABSCOTable(sigma, table_axes)
    end
end

"""Name of the `Gas_XX_Absorption` dataset to read, from `gas` or the only one present."""
function absco_gas_dataset(file, gas, path)
    if gas !== nothing
        name = string("Gas_", lpad(gas, 2, '0'), "_Absorption")
        haskey(file, name) || throw(ArgumentError(
            "ABSCO file $path has no dataset \"$name\"; datasets: $(keys(file))"))
        return name
    end
    names = filter(n -> occursin(r"^Gas_\d+_Absorption$", n), keys(file))
    length(names) == 1 || throw(ArgumentError(
        "ABSCO file $path holds $(length(names)) Gas_XX_Absorption datasets ($names); pass gas = n"))
    return only(names)
end

"""Contiguous index range of the wavenumbers inside `range_cm`, plus one neighbour each side."""
function absco_wavenumber_indices(nu::Vector{Float64}, range_cm, path)
    range_cm === nothing && return 1:length(nu)
    lo, hi = Float64(range_cm[1]), Float64(range_cm[2])
    lo < hi || throw(ArgumentError("range_cm must be increasing, got ($lo, $hi)"))
    (nu[1] <= lo && hi <= nu[end]) || throw(ArgumentError(
        "range_cm ($lo, $hi) is outside the ABSCO wavenumbers ($(nu[1]), $(nu[end])) in $path"))
    first_index = max(searchsortedlast(nu, lo), 1)
    last_index = min(searchsortedfirst(nu, hi), length(nu))
    last_index > first_index || (last_index = min(first_index + 1, length(nu)))
    return first_index:last_index
end

export read_absco_table
