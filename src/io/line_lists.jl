# Loaders for cross-section models: HITRAN line lists (`HitranVoigt`) and vSmartMOM
# interpolation tables saved with JLD2 (`LookupTable`). Replaces v0.1 `get_molecule_info`,
# `load_interp_model` and `calculate_cross_sections`.

"""
    load_cross_section_model(HitranVoigt, par_path; molecule, isotope = -1, range_cm,
                             line_margin_cm = 0.0, wing_cutoff_cm = 40,
                             broadening = Absorption.Voigt(), self_vmr = 0.0,
                             error_function = Absorption.HumlicekWeidemann32SDErrorFunction(),
                             min_strength = 0.0) -> HitranVoigt

Read the lines of HITRAN molecule number `molecule` (6 for CH4) and isotopologue `isotope`
(-1 for all) from the 160-character `.par` file `par_path`, with line centres in
`range_cm = (minimum, maximum)` widened by `line_margin_cm` on each side (cm-1), and build a
line-by-line `HitranVoigt` model valid over `range_cm`.

vSmartMOM options: `wing_cutoff_cm` (integer, cm-1) truncates each line, `broadening` picks
the line shape (`Voigt()`, `Lorentz()`, `Doppler()`), `self_vmr` (mol/mol) is the
self-broadening mole fraction, `error_function` is the complex error function, and lines with
strength below `min_strength` (cm/molecule) are skipped. The defaults reproduce v0.1, which
read lines only inside the grid (`line_margin_cm = 0`).

```julia
ch4 = load_cross_section_model(HitranVoigt, "hit20.par"; molecule = 6, isotope = 1,
                               range_cm = (6076.0, 6084.0))
```
"""
function load_cross_section_model(::Type{HitranVoigt}, par_path::AbstractString;
                                  molecule::Integer, isotope::Integer = -1,
                                  range_cm::Tuple{Real,Real}, line_margin_cm::Real = 0.0,
                                  wing_cutoff_cm::Integer = 40,
                                  broadening = Absorption.Voigt(), self_vmr::Real = 0.0,
                                  error_function = Absorption.HumlicekWeidemann32SDErrorFunction(),
                                  min_strength::Real = 0.0)
    isfile(par_path) || throw(ArgumentError("HITRAN line list not found: $par_path"))
    lo, hi = Float64(range_cm[1]), Float64(range_cm[2])
    lo < hi || throw(ArgumentError("range_cm must be increasing, got ($lo, $hi)"))
    line_margin_cm >= 0 || throw(ArgumentError("line_margin_cm must be >= 0, got $line_margin_cm"))
    table = try
        Absorption.read_hitran(par_path; mol = Int(molecule), iso = Int(isotope),
                               ν_min = lo - line_margin_cm, ν_max = hi + line_margin_cm,
                               min_strength = min_strength)
    catch err
        err isa Absorption.HitranEmptyError || rethrow()
        throw(ArgumentError("no lines of molecule $molecule, isotope $isotope between " *
                            "$(lo - line_margin_cm) and $(hi + line_margin_cm) cm-1 in $par_path"))
    end
    model = Absorption.make_hitran_model(table, broadening; wing_cutoff = wing_cutoff_cm,
                                         vmr = self_vmr, CEF = error_function,
                                         architecture = vSmartMOM.Architectures.CPU())
    return HitranVoigt(model; range_cm = (lo, hi))
end

"""
    load_cross_section_model(LookupTable, jld2_path; key = "itp_model") -> LookupTable

Load a vSmartMOM `InterpolationModel` saved with JLD2 (by `save_interpolation_model` or
`@save path itp_model`) and wrap it as a `LookupTable` over wavenumber (cm-1), pressure (hPa)
and temperature (K).

```julia
ch4 = load_cross_section_model(LookupTable, "merlin_12CH4.jld2")
```
"""
function load_cross_section_model(::Type{LookupTable}, jld2_path::AbstractString;
                                  key::AbstractString = "itp_model")
    isfile(jld2_path) || throw(ArgumentError("lookup table file not found: $jld2_path"))
    model = JLD2.jldopen(jld2_path, "r") do file
        haskey(file, key) || throw(ArgumentError(
            "lookup table file $jld2_path has no entry \"$key\"; entries: $(keys(file))"))
        file[key]
    end
    model isa Absorption.InterpolationModel || throw(ArgumentError(
        "entry \"$key\" in $jld2_path is a $(typeof(model)), not a vSmartMOM InterpolationModel"))
    return LookupTable(model)
end

export load_cross_section_model
