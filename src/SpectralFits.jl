"""
    SpectralFits

Optimal-estimation retrievals of trace gases from spectral measurements (dual-comb, FTS,
grating and open-path wavelength-modulation spectroscopy).

Layers, with dependencies pointing inward only:

- `SpectralFits.CoreLayer`: pure science (cross sections, atmospheres, instruments, baselines,
  forward model, solvers, diagnostics). No file I/O, no printing, no global state.
- `SpectralFits.IOLayer`: every file read and write.
- `SpectralFits.Apps`: workflows such as `Apps.BatchRetrieval`.

The layer modules are not called `Core` and `IO` because those names would shadow Julia's
own `Core` module and `Base.IO` type inside the package.

See `ARCHITECTURE.md` and `docs/architecture_plan.md`.
"""
module SpectralFits

include("core/CoreLayer.jl")
include("io/IOLayer.jl")
include("apps/Apps.jl")

using .CoreLayer
using .IOLayer

# re-export the public API of core and io
for layer in (CoreLayer, IOLayer), name in names(layer)
    name === nameof(layer) && continue
    @eval export $name
end

end # module
