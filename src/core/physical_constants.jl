# Physical constants: CODATA 2018 exact SI values and standard gravity (open question 2 in
# docs/architecture_plan.md, resolved). v0.1 used R = 8.314472, N_A = 6.0221415e23 and
# g = 9.8196; the v0.1 regression reference was regenerated with these new values patched
# into the legacy code (test/regression/README.md), so parity is kept exactly.

"""Speed of light in cm/s."""
const SPEED_OF_LIGHT_CM_PER_S = 299792458.0 * 100.0
"""Avogadro constant in 1/mol (CODATA 2018, exact)."""
const AVOGADRO_PER_MOL = 6.02214076e23
"""Boltzmann constant in J/K (CODATA 2018, exact)."""
const BOLTZMANN_J_PER_K = 1.380649e-23
"""Universal gas constant in J/(mol K): N_A * k_B = 8.314462618... (CODATA 2018, exact)."""
const GAS_CONSTANT_J_PER_MOL_K = AVOGADRO_PER_MOL * BOLTZMANN_J_PER_K
"""Standard gravity in m/s2 (CGPM 1901, exact) used for hydrostatic columns."""
const GRAVITY_M_PER_S2 = 9.80665
"""Molar mass of dry air in kg/mol."""
const MOLAR_MASS_DRY_AIR_KG_PER_MOL = 28.9647e-3
"""Molar mass of water in kg/mol."""
const MOLAR_MASS_H2O_KG_PER_MOL = 18.01528e-3

export SPEED_OF_LIGHT_CM_PER_S, GAS_CONSTANT_J_PER_MOL_K, AVOGADRO_PER_MOL, BOLTZMANN_J_PER_K
export GRAVITY_M_PER_S2
export MOLAR_MASS_DRY_AIR_KG_PER_MOL, MOLAR_MASS_H2O_KG_PER_MOL
