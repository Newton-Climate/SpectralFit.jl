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

## Features
 Users can customize forward model for radiation and instrument simulations
- Applies Bayesian Maximum Likely-hood Estimation with custom priors and regularization hyper-parameters
- Both built-in and custom molecular absorption models that can run on both GPU and CPU
- Vertical column retrieval to calculate variations of gases with height
- Support for multiple instrument operators 
- Post-processing tools to calculate information content 

