# Air number densities and columns (section 3.5). Pressure in hPa, temperature in K, path
# length in cm, columns in molecules per cm2.

"""
    number_density_per_cm3(pressure_hPa, temperature_K)

Ideal-gas number density of air in molecules per cm3.

```julia
number_density_per_cm3(1013.25, 273.15)   # about 2.687e19, the Loschmidt number
```
"""
number_density_per_cm3(pressure_hPa, temperature_K) =
    pressure_hPa * 100 / (GAS_CONSTANT_J_PER_MOL_K * temperature_K) * AVOGADRO_PER_MOL / 1e6

"""
    homogeneous_air_column(pressure_hPa, temperature_K, path_length_cm)

Total (wet) air column along a homogeneous path, molecules per cm2: number density times path
length. Equals the v0.1 `calc_vcd(p, T, path_length)`.
"""
homogeneous_air_column(pressure_hPa, temperature_K, path_length_cm) =
    number_density_per_cm3(pressure_hPa, temperature_K) * path_length_cm

"""
    hydrostatic_dry_air_column(delta_p_hPa)

Dry-air column of a layer of pressure thickness `delta_p_hPa` in hydrostatic balance,
molecules per cm2: `delta_p / (m_dry * g)` with `m_dry` the mass of one dry-air molecule.
"""
function hydrostatic_dry_air_column(delta_p_hPa)
    molecule_mass_kg = MOLAR_MASS_DRY_AIR_KG_PER_MOL / AVOGADRO_PER_MOL
    return delta_p_hPa * 100 / (molecule_mass_kg * GRAVITY_M_PER_S2) / 1e4
end

"""
    layer_pressure_thickness_hPa(pressure_hPa::AbstractVector)

Pressure thickness of each layer from layer-centre pressures: half levels are midpoints
between neighbouring centres, and the outer half levels mirror the nearest interior one.
Same as v0.1 `half_pressure_levels`, without the equal-end-pressure special case of
`make_vcd_profile`. Needs at least two layers.
"""
function layer_pressure_thickness_hPa(p::AbstractVector)
    n = length(p)
    n >= 2 || throw(ArgumentError("pressure_hPa: need at least two layers, got $n"))
    half = similar(p, n + 1)
    for i in 2:n
        half[i] = (p[i] + p[i-1]) / 2
    end
    half[1] = p[1] - (half[2] - p[1])
    half[n+1] = p[n] + (p[n] - half[n])
    return abs.(diff(half))
end

export number_density_per_cm3, homogeneous_air_column, hydrostatic_dry_air_column
export layer_pressure_thickness_hPa
