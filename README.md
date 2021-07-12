# SpectralFits.jl

SpectralFits.jl is a modern and flexible spectral fitting routine for atmospheric remote sensing. It has support for multiple species and spectral line-lists, including HiTran, TCCON, and OCO, and can fit an arbitrary number of species. We have plans to add more instrument operators to the package as well.

## Installation Instructions

```
using Pkg
Pkg.add(url="https://github.com/Newton-Climate/SpectralFit.jl.git")
using SpectralFits
```

## Example Run
Please see the example_run.jl script for how to run the code

## Future Plans
- Create Vertical column retrieval to calculate variations of gases with height
- Add support for multiple instrument operators 
- Organize into modules for pre-compiling speedups
