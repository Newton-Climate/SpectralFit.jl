# Phase 0 regression reference for the SpectralFits.jl re-architecture.
#
# Run ONCE against the tag v0.1-legacy (the old API) with three constants in the legacy
# src/constants.jl patched to the values the new code uses (CODATA 2018 / standard gravity):
#     const r  = 6.02214076e23 * 1.380649e-23   # N_A * k_B   (was 8.314472)
#     const Nₐ = 6.02214076e23                  # (was 6.0221415e23)
#     const g₀ = 9.80665                        # standard gravity (was 9.8196)
# Nothing else in the legacy code is changed. The script asserts these values are loaded.
# It writes reference_v0_1.jld2, which holds only plain numbers, strings, Bools and
# Float64/Int arrays, so it loads with `using JLD2` alone. See README.md for the meaning
# of every key and for how the environment is built.
#
# Usage (from an environment where the patched v0.1-legacy SpectralFits is installed):
#     julia --project=<env> generate_reference_v0_1.jl [output.jld2]
#
# Configurations recorded:
#   ch4_fm_*  : single-layer forward model, CH4 window, intensity space (linear=false)
#               and log space (linear=true), column units (fit_column=true).
#   co2_fm_*  : same for the CO2 window.
#   ch4_inv   : nonlinear_inversion, column units, intensity space (linear=false),
#               no prior, default Se, synthetic noisy measurement with a fixed seed.
#   prof_inv  : profile_inversion on a 3-layer synthetic case (state and chi squared only
#               are regression targets; the posterior covariance is NOT recorded, issue 1).

using SpectralFits, vSmartMOM, vSmartMOM.Absorption
using LinearAlgebra, Random, Dates, ForwardDiff, JLD2, Pkg

const SF = SpectralFits
const OUT = length(ARGS) >= 1 ? ARGS[1] : joinpath(@__DIR__, "reference_v0_1.jld2")
const PAR = joinpath(pkgdir(SpectralFits), "test", "data", "hit20_subset.par")
@assert isfile(PAR) "line list not found: $PAR"

R = Dict{String,Any}()   # everything that goes into the file; keys are flat "group/name"

# ---------------------------------------------------------------- metadata and constants
pkgver(name) = string(Pkg.dependencies()[findfirst(d -> d.name == name, Pkg.dependencies())].version)
R["meta/description"] = "SpectralFits v0.1-legacy regression reference (Phase 0), constants patched to CODATA 2018 / standard gravity. See test/regression/README.md."
R["meta/legacy_tag"] = "v0.1-legacy"
R["meta/julia_version"] = string(VERSION)
R["meta/vSmartMOM_version"] = pkgver("vSmartMOM")
R["meta/Interpolations_version"] = pkgver("Interpolations")
R["meta/ForwardDiff_version"] = pkgver("ForwardDiff")
R["meta/generated_utc"] = string(Dates.now(Dates.UTC))
R["meta/line_list_file"] = "test/data/hit20_subset.par"
R["meta/constants_patch"] = "v0.1-legacy src/constants.jl with r = N_A*k_B (CODATA 2018), " *
                            "N_A = 6.02214076e23, g0 = 9.80665; nothing else modified"

@assert SF.r == 6.02214076e23 * 1.380649e-23 "legacy constants not patched (r = $(SF.r))"
@assert SF.Nₐ == 6.02214076e23 "legacy constants not patched (N_A = $(SF.Nₐ))"
@assert SF.g₀ == 9.80665 "legacy constants not patched (g0 = $(SF.g₀))"

R["const/c_cm_per_s"] = Float64(SF.c)
R["const/R_J_per_mol_K"] = SF.r
R["const/N_A_per_mol"] = SF.Nₐ
R["const/g0_m_per_s2"] = SF.g₀
R["const/dry_air_molar_mass_kg_per_mol"] = 28.9647e-3
R["const/h2o_molar_mass_kg_per_mol"] = 18.01528e-3

# cross-section model settings (vSmartMOM make_hitran_model defaults used by get_molecule_info)
R["xsec_model/type"] = "vSmartMOM.Absorption.HitranModel"
R["xsec_model/broadening"] = "Voigt"
R["xsec_model/wing_cutoff_cm-1"] = 40
R["xsec_model/vmr"] = 0.0
R["xsec_model/CEF"] = "HumlicekWeidemann32SDErrorFunction"
R["xsec_model/architecture"] = "CPU"
R["xsec_model/line_selection"] = "read_hitran(file, mol, iso, ν_min=fine_grid[1], ν_max=fine_grid[end])"

# ---------------------------------------------------------------- helpers
const PATHLENGTH = 195017.0     # cm (calc_vcd treats it as cm; see plan issue 9)

function store_grid!(prefix, g::AbstractRange)
    R["$prefix/fine_grid_start"] = Float64(first(g))
    R["$prefix/fine_grid_step"] = Float64(step(g))
    R["$prefix/fine_grid_stop"] = Float64(last(g))
    R["$prefix/fine_grid_length"] = length(g)
    R["$prefix/fine_grid"] = collect(Float64, g)
end

function store_lines!(prefix, info::MolecularMetaData)
    h = info.model.hitran
    R["$prefix/mol"] = info.molecule_num
    R["$prefix/iso"] = info.isotope_num
    for fn in fieldnames(typeof(h))
        v = getfield(h, fn)
        if v isa AbstractVector{<:Real}
            R["$prefix/lines/$(fn)"] = collect(Float64, v)
        elseif v isa AbstractVector{<:Integer}
            R["$prefix/lines/$(fn)"] = collect(Int, v)
        end
    end
    R["$prefix/n_lines"] = length(h.νᵢ)
end

function make_meas(intensity, grid; p, T, σ², vcd)
    FrequencyCombMeasurement(intensity=collect(float.(intensity)), grid=grid,
                             temperature=p isa AbstractVector ? T : float(T),
                             pressure=p isa AbstractVector ? p : float(p),
                             time=DateTime(2020, 1, 1), pathlength=PATHLENGTH, vcd=vcd,
                             num_averaged_measurements=1, averaging_window=Second(1),
                             machine_time=0.0, σ²=σ²)
end

"""Forward model in one window: record inputs, intermediates and outputs for linear=false/true."""
function record_window!(tag, fine_grid, meas_grid, gases::Vector{Tuple{String,Int,Int,Float64}};
                        p, T, poly_degree, shape_intensity, shape_log)
    infos = [get_molecule_info(name, PAR, mol, iso, fine_grid) for (name, mol, iso, _) in gases]
    spectra = setup_molecules(infos)
    vcd = SF.calc_vcd(p, T, PATHLENGTH)

    store_grid!("$tag", fine_grid)
    R["$tag/meas_grid"] = collect(Float64, meas_grid)
    R["$tag/pressure_hPa"] = p
    R["$tag/temperature_K"] = T
    R["$tag/pathlength_cm"] = PATHLENGTH
    R["$tag/vcd_air_molec_per_cm2"] = vcd
    R["$tag/molecules"] = [g[1] for g in gases]
    for (k, (name, mol, iso, vmr)) in enumerate(gases)
        store_lines!("$tag/mol_$(name)", infos[k])
        R["$tag/mol_$(name)/vmr_true"] = vmr
        σ = absorption_cross_section(infos[k].model, fine_grid, p, T)
        R["$tag/xsec_$(name)_cm2_per_molec"] = collect(Float64, σ)
    end

    for (space, linear, shape) in (("intensity", false, shape_intensity), ("log", true, shape_log))
        pre = "$(tag)_$(space)"
        x = StateVector([name => vmr * vcd for (name, _, _, vmr) in gases]...,
                        "shape_parameters" => shape)
        setup = RetrievalSetup("poly_degree" => poly_degree, "fit_column" => true,
                               "linear" => linear, "verbose_mode" => false)
        meas = make_meas(zeros(length(meas_grid)), meas_grid; p=p, T=T, σ²=1e-8, vcd=vcd)
        f = generate_forward_model(x, meas, spectra, setup)
        y = f(x)
        yv = f(assemble_state_vector!(x))
        @assert y == yv

        # intermediates (re-derived with the same legacy functions)
        trans_fine = SF.calculate_transmission(x, spectra, PATHLENGTH, length(fine_grid); p=p, T=T, fit_column=true)
        trans_meas = SF.apply_instrument(fine_grid, trans_fine, meas_grid)
        baseline = vec(calc_polynomial_term(poly_degree, shape, length(meas_grid)))
        y_check = linear ? log.(trans_meas) .+ baseline : trans_meas .* baseline
        @assert y_check == y "intermediate reconstruction differs from forward model"

        R["$pre/state_names"] = [collect(keys(x))[1:end-1]..., ["shape_parameters[$i]" for i in 1:poly_degree]...]
        R["$pre/state_vector"] = collect(Float64, assemble_state_vector!(x))
        for (name, _, _, _) in gases
            R["$pre/state_$(name)_molec_per_cm2"] = x[name]
        end
        R["$pre/shape_parameters"] = collect(Float64, shape)
        R["$pre/setup_poly_degree"] = poly_degree
        R["$pre/setup_fit_column"] = true
        R["$pre/setup_linear"] = linear
        R["$pre/setup_use_OCO_table"] = false
        R["$pre/setup_adjust_ch4_broadening"] = false
        R["$pre/pT_in_state"] = false
        R["$pre/transmission_fine"] = collect(Float64, trans_fine)
        R["$pre/transmission_meas"] = collect(Float64, trans_meas)
        R["$pre/baseline"] = collect(Float64, baseline)
        R["$pre/spectrum"] = collect(Float64, y)
    end

    # p and T as state elements (non-cached path) must give the same spectrum at the same values
    x = StateVector([name => vmr * vcd for (name, _, _, vmr) in gases]...,
                    "pressure" => p, "temperature" => T, "shape_parameters" => shape_intensity)
    setup = RetrievalSetup("poly_degree" => poly_degree, "fit_column" => true, "linear" => false, "verbose_mode" => false)
    meas = make_meas(zeros(length(meas_grid)), meas_grid; p=p, T=T, σ²=1e-8, vcd=vcd)
    ypt = generate_forward_model(x, meas, spectra, setup)(x)
    R["$(tag)_intensity/spectrum_pT_in_state_maxabsdiff"] = maximum(abs, ypt .- R["$(tag)_intensity/spectrum"])
    return spectra, vcd
end

# ---------------------------------------------------------------- 1. forward models
const CH4_FINE = 6077.0:0.01:6083.0
const CH4_MEAS = collect(6078.0:0.0067:6082.0)
const CO2_FINE = 6225.0:0.01:6231.0
const CO2_MEAS = collect(6226.0:0.0067:6230.0)
const P0, T0 = 850.0, 290.0
const POLY = 4
const SHAPE_I = [1.0, 0.02, -0.01, 0.005]   # multiplicative (intensity space)
const SHAPE_L = [0.0, 0.02, -0.01, 0.005]   # additive to log transmission (log space)

ch4_gases = [("H2O", 1, 1, 0.01), ("CH4", 6, 1, 1900e-9)]
co2_gases = [("H2O", 1, 1, 0.01), ("CO2", 2, 1, 400e-6)]

ch4_spectra, ch4_vcd = record_window!("ch4_fm", CH4_FINE, CH4_MEAS, ch4_gases;
                                      p=P0, T=T0, poly_degree=POLY, shape_intensity=SHAPE_I, shape_log=SHAPE_L)
record_window!("co2_fm", CO2_FINE, CO2_MEAS, co2_gases;
               p=P0, T=T0, poly_degree=POLY, shape_intensity=SHAPE_I, shape_log=SHAPE_L)

# ---------------------------------------------------------------- 2. nonlinear_inversion (CH4)
let
    σ = 1e-4
    seed = 3   # seed 42 gave a 3.1-sigma CH4 draw; seed 3 is a typical draw (see README)
    vcd = ch4_vcd
    x_true = StateVector("H2O" => 0.01 * vcd, "CH4" => 1900e-9 * vcd, "shape_parameters" => SHAPE_I)
    setup = RetrievalSetup("poly_degree" => POLY, "fit_column" => true, "linear" => false, "verbose_mode" => false)
    template = make_meas(zeros(length(CH4_MEAS)), CH4_MEAS; p=P0, T=T0, σ²=σ^2, vcd=vcd)
    y_clean = generate_forward_model(x_true, template, ch4_spectra, setup)(x_true)
    noise = σ .* randn(MersenneTwister(seed), length(CH4_MEAS))
    y = y_clean .+ noise
    meas = make_meas(y, CH4_MEAS; p=P0, T=T0, σ²=σ^2, vcd=vcd)

    xa = StateVector("H2O" => 0.012 * vcd, "CH4" => 1700e-9 * vcd,
                     "shape_parameters" => [maximum(y), 0.0, 0.0, 0.0])
    f = generate_forward_model(xa, meas, ch4_spectra, setup)
    iterates = Vector{Vector{Float64}}()
    fw(x) = (eltype(x) <: ForwardDiff.Dual && push!(iterates, ForwardDiff.value.(x)); f(x))
    res = nonlinear_inversion(fw, xa, meas, ch4_spectra, setup)
    @assert res isa InversionResults

    P = "ch4_inv"
    R["$P/window"] = "ch4_fm (same fine grid, meas grid, p, T, path, cross-sections)"
    R["$P/state_names"] = ["H2O", "CH4", ["shape_parameters[$i]" for i in 1:POLY]...]
    R["$P/x_true"] = collect(Float64, assemble_state_vector!(x_true))
    R["$P/x_first_guess"] = collect(Float64, assemble_state_vector!(xa))
    R["$P/noise_sigma"] = σ
    R["$P/noise_seed_MersenneTwister"] = seed
    R["$P/noise"] = noise
    R["$P/y_clean"] = collect(Float64, y_clean)
    R["$P/y_measured"] = collect(Float64, y)
    R["$P/setup_poly_degree"] = POLY
    R["$P/setup_fit_column"] = true
    R["$P/setup_linear"] = false
    R["$P/setup_prior"] = "none"
    R["$P/setup_obs_covariance"] = "default: make_obs_error(meas, linear=false) = Diagonal(1/sigma^2)"
    R["$P/tolerance"] = 1.0e-4
    R["$P/max_iter"] = 30
    R["$P/x_retrieved"] = collect(Float64, assemble_state_vector!(res.x))
    R["$P/chi2"] = res.χ²
    R["$P/S"] = Matrix{Float64}(res.S)
    R["$P/K"] = Matrix{Float64}(res.K)
    R["$P/model"] = collect(Float64, res.model)
    R["$P/model_at_x_retrieved"] = collect(Float64, f(res.x))
    R["$P/Se_inv_diag"] = collect(Float64, diag(res.Sₑ⁻¹))
    R["$P/n_iterations"] = length(iterates)
    R["$P/iterates"] = reduce(hcat, iterates)   # state_length x n_iterations: states where K was evaluated
end

# ---------------------------------------------------------------- 3. profile_inversion
let
    σ = 1e-4
    seed = 11
    sza = 30.0
    p = [850.0, 650.0, 450.0]
    T = [290.0, 275.0, 255.0]
    nl = length(p)
    poly = 3
    vcd = make_vcd_profile(p, T)                     # molec/cm2 per layer (hydrostatic)
    spectra = ch4_spectra
    mk(h2o, ch4, shape) = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => h2o, "CH4" => ch4, "shape_parameters" => shape)
    x_true = mk(0.005 .* vcd, [1950e-9, 1880e-9, 1820e-9] .* vcd, [1.0, 0.01, -0.005])
    xa = mk(0.006 .* vcd, fill(1800e-9, nl) .* vcd, [1.0, 0.0, 0.0])
    σa = mk(0.005 .* vcd, fill(300e-9, nl) .* vcd, [1.0, 1.0, 1.0])
    γ = 0.0
    setup = RetrievalSetup("poly_degree" => poly, "fit_column" => true, "linear" => false,
                           "verbose_mode" => false, "sza" => sza, "σ" => σa, "γ" => γ)

    template = make_meas(zeros(length(CH4_MEAS)), CH4_MEAS; p=p, T=T, σ²=σ^2, vcd=vcd)
    ftrue = generate_profile_model(x_true, template, spectra, setup)
    y_clean = ftrue(assemble_state_vector!(x_true))
    noise = σ .* randn(MersenneTwister(seed), length(CH4_MEAS))
    y = y_clean .+ noise
    meas = make_meas(y, CH4_MEAS; p=p, T=T, σ²=σ^2, vcd=vcd)
    xa["shape_parameters"] = [maximum(y), 0.0, 0.0]

    f = generate_profile_model(xa, meas, spectra, setup)
    iterates = Vector{Vector{Float64}}()
    fw(x) = (eltype(x) <: ForwardDiff.Dual && push!(iterates, ForwardDiff.value.(x)); f(x))
    res = profile_inversion(fw, xa, meas, spectra, setup)

    P = "prof_inv"
    R["$P/n_layers"] = nl
    R["$P/pressure_hPa"] = p
    R["$P/temperature_K"] = T
    R["$P/vcd_layer_molec_per_cm2"] = collect(Float64, vcd)
    R["$P/sza_deg"] = sza
    R["$P/air_mass_factor"] = 1.0 / cosd(sza)
    R["$P/fine_grid_start"] = Float64(first(CH4_FINE)); R["$P/fine_grid_step"] = Float64(step(CH4_FINE))
    R["$P/fine_grid_stop"] = Float64(last(CH4_FINE)); R["$P/fine_grid"] = collect(Float64, CH4_FINE)
    R["$P/fine_grid_times_shift_equals_grid"] = collect(1.0 .* CH4_FINE) == collect(CH4_FINE)
    R["$P/meas_grid"] = collect(Float64, CH4_MEAS)
    R["$P/molecules"] = ["H2O", "CH4"]
    R["$P/state_names"] = [["H2O[$i]" for i in 1:nl]..., ["CH4[$i]" for i in 1:nl]...,
                           ["shape_parameters[$i]" for i in 1:poly]...]
    R["$P/x_true"] = collect(Float64, assemble_state_vector!(x_true))
    R["$P/x_prior"] = collect(Float64, assemble_state_vector!(xa))   # also the first guess
    R["$P/prior_sigma"] = collect(Float64, assemble_state_vector!(σa))
    R["$P/Sa_inv_diag"] = collect(Float64, (1 ./ assemble_state_vector!(σa)) .^ 2)
    R["$P/gamma"] = γ
    R["$P/noise_sigma"] = σ
    R["$P/noise_seed_MersenneTwister"] = seed
    R["$P/noise"] = noise
    R["$P/y_clean"] = collect(Float64, y_clean)
    R["$P/y_measured"] = collect(Float64, y)
    R["$P/setup_poly_degree"] = poly
    R["$P/setup_fit_column"] = true
    R["$P/setup_linear"] = false
    R["$P/setup_obs_covariance"] = "default: make_obs_error(meas, linear=false) = Diagonal(1/sigma^2)"
    R["$P/tolerance"] = 1.0e-6
    R["$P/max_iter"] = 30
    R["$P/x_retrieved"] = collect(Float64, assemble_state_vector!(res.x))
    R["$P/chi2"] = res.χ²
    R["$P/model"] = collect(Float64, res.model)
    R["$P/n_iterations"] = length(iterates)
    R["$P/iterates"] = reduce(hcat, iterates)
    for name in ("H2O", "CH4")
        R["$P/xsec_$(name)_cm2_per_molec"] = reduce(hcat, [collect(Float64,
            absorption_cross_section(spectra[name].model, CH4_FINE, p[i], T[i])) for i in 1:nl])
    end
end

# ---------------------------------------------------------------- write
for (k, v) in R   # guard: only plain data
    ok = v isa Union{Number,String,Bool} || v isa Array{<:Union{Float64,Int,String,Bool}}
    ok || error("non-plain value for key $k: $(typeof(v))")
end
jldopen(OUT, "w") do file
    for k in sort(collect(keys(R)))
        file[k] = R[k]
    end
end
println("wrote $(length(R)) keys to $OUT")
println("ch4_inv chi2 = ", R["ch4_inv/chi2"], "  iterations = ", R["ch4_inv/n_iterations"])
println("ch4_inv x_retrieved = ", R["ch4_inv/x_retrieved"])
println("prof_inv chi2 = ", R["prof_inv/chi2"], "  iterations = ", R["prof_inv/n_iterations"])
println("pT-in-state max abs diff ch4/co2 = ", R["ch4_fm_intensity/spectrum_pT_in_state_maxabsdiff"], " ",
        R["co2_fm_intensity/spectrum_pT_in_state_maxabsdiff"])
