# Physical constants. Values match the v0.1 code for regression parity
# (open question 2 in docs/architecture_plan.md).

"""Speed of light in cm/s."""
const SPEED_OF_LIGHT_CM_PER_S = 299792458.0 * 100.0
"""Universal gas constant in J/(mol K)."""
const GAS_CONSTANT_J_PER_MOL_K = 8.314472
"""Avogadro constant in 1/mol."""
const AVOGADRO_PER_MOL = 6.0221415e23
"""Gravitational acceleration in m/s2 used for hydrostatic columns."""
const GRAVITY_M_PER_S2 = 9.8196
"""Molar mass of dry air in kg/mol."""
const MOLAR_MASS_DRY_AIR_KG_PER_MOL = 28.9647e-3
"""Molar mass of water in kg/mol."""
const MOLAR_MASS_H2O_KG_PER_MOL = 18.01528e-3

export SPEED_OF_LIGHT_CM_PER_S, GAS_CONSTANT_J_PER_MOL_K, AVOGADRO_PER_MOL, GRAVITY_M_PER_S2
export MOLAR_MASS_DRY_AIR_KG_PER_MOL, MOLAR_MASS_H2O_KG_PER_MOL
