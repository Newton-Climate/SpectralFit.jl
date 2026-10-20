# Lookup tables built from any cross-section model: tabulate on a (wavenumber, pressure,
# temperature) grid and fit the cubic B-spline that LookupTable interpolates.

"""
    tabulate_cross_sections(model, wavenumber_cm, pressure_hPa, temperature_K;
                            h2o_dry_mole_fraction = 0.0) -> Array{Float64,3}

Cross sections of `model` (cm2/molecule) on every node of the grid, indexed
`[wavenumber, pressure, temperature]`. Pressure (hPa) and temperature (K) nodes are
evaluated in parallel over Julia threads (start Julia with `-t auto` to use them).

```julia
cs = tabulate_cross_sections(ch4, 6076.0:0.001:6078.0, 600.0:10.0:1050.0, 230.0:10.0:350.0)
```
"""
function tabulate_cross_sections(model::AbstractCrossSectionModel, wavenumber_cm::AbstractRange,
                                 pressure_hPa::AbstractRange, temperature_K::AbstractRange;
                                 h2o_dry_mole_fraction::Real = 0.0)
    lo, hi = spectral_range_cm(model)
    (lo <= first(wavenumber_cm) && last(wavenumber_cm) <= hi) || throw(ArgumentError(
        "tabulate_cross_sections: wavenumber grid $(first(wavenumber_cm))-$(last(wavenumber_cm)) cm-1 " *
        "is outside the model range $lo-$hi cm-1"))
    cs = zeros(length(wavenumber_cm), length(pressure_hPa), length(temperature_K))
    nodes = [(i, j) for i in eachindex(pressure_hPa), j in eachindex(temperature_K)]
    Threads.@threads for (i, j) in nodes
        c = GasConditions(Float64(pressure_hPa[i]), Float64(temperature_K[j]), Float64(h2o_dry_mole_fraction))
        cs[:, i, j] .= cross_section(model, wavenumber_cm, c)
    end
    return cs
end

"""
    lookup_table_from_grid(cross_sections, wavenumber_cm, pressure_hPa, temperature_K;
                           molecule = -1, isotope = -1, coefficient_type = Float64) -> LookupTable

`LookupTable` from cross sections tabulated on the grid (`tabulate_cross_sections`): a cubic
B-spline with line boundary conditions on the grid nodes, as vSmartMOM's
`make_interpolation_model`. `coefficient_type = Float32` halves the size; `molecule` and
`isotope` (HITRAN numbers) are stored with the table.
"""
function lookup_table_from_grid(cs::AbstractArray{<:Real,3}, wavenumber_cm::AbstractRange,
                                pressure_hPa::AbstractRange, temperature_K::AbstractRange;
                                molecule::Integer = -1, isotope::Integer = -1,
                                coefficient_type::Type{<:AbstractFloat} = Float64)
    size(cs) == (length(wavenumber_cm), length(pressure_hPa), length(temperature_K)) || throw(DimensionMismatch(
        "lookup_table_from_grid: cross sections of size $(size(cs)) for a grid of " *
        "$((length(wavenumber_cm), length(pressure_hPa), length(temperature_K)))"))
    all(isfinite, cs) || throw(ArgumentError("lookup_table_from_grid: cross sections are not all finite"))
    itp = Interpolations.interpolate(coefficient_type.(cs),
                                     Interpolations.BSpline(Interpolations.Cubic(Interpolations.Line(Interpolations.OnGrid()))))
    model = Absorption.InterpolationModel(itp, Int(molecule), Int(isotope), wavenumber_cm,
                                          pressure_hPa, temperature_K)
    return LookupTable(model)
end

export tabulate_cross_sections, lookup_table_from_grid
