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

## Running the tests
The package targets Julia 1.10+ (developed on 1.12). From the repository root:

```
julia --project=. -e 'using Pkg; Pkg.test()'
```

Unit tests cover `src/utils.jl`, `src/read_data.jl` and `src/spectroscopy.jl`; integration tests
simulate spectra with the forward model and check that the retrievals (single-layer, profile,
multi-window, and the full HDF5 → JLD2 pipeline) recover the true state. They use the small
HITRAN 2020 subset in `test/data/hit20_subset.par`, so no external line lists are needed.

## Features
 Users can customize forward model for radiation and instrument simulations
- Applies Bayesian Maximum Likely-hood Estimation with custom priors and regularization hyper-parameters
- Both built-in and custom molecular absorption models that can run on both GPU and CPU
- Vertical column retrieval to calculate variations of gases with height
- Support for multiple instrument operators 
- Post-processing tools to calculate information content 

