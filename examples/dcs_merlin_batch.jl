# Worked example: the MERLIN CH4 R(6) window retrieval of
# DCS_experiments/DCS_retrievals/run_merlin.jl, ported to the v0.2 API (section 9 of
# docs/architecture_plan.md). Run from DCS_experiments/DCS_retrievals with
#     julia --project=<SpectralFit.jl>/test <SpectralFit.jl>/examples/dcs_merlin_batch.jl
#
# Lookup tables written by older Julia/JLD2 versions may not load under Julia 1.12 (their
# StepRangeLen axes do not reconstruct); regenerate them, or use HitranVoigt line lists:
#     ch4 = load_cross_section_model(HitranVoigt, "../spectra/hit20_12CH4.par"; molecule = 6,
#                                    isotope = 1, range_cm = (6075.0, 6079.0), line_margin_cm = 1.0)

using SpectralFits
using SpectralFits.Apps.BatchRetrieval: BatchRetrievalConfig, run
using Dates, Distributed

ch4 = load_cross_section_model(LookupTable, "../spectra/merlin_12CH4.jld2")
h2o = load_cross_section_model(LookupTable, "../spectra/hit20_H2O_narrow.jld2")
absorbers = (H2O = Absorber(:H2O, h2o), CH4 = Absorber(:CH4, ch4))

retrieval = RetrievalConfig(
    windows_cm = [(6076.05, 6077.95)],
    baseline_coefficients = [5],
    state_units = ColumnDensity(),
    measurement_space = LinearSpace(),
    instrument = DCS(),
    fine_grid_spacing_cm = 0.003,
    solver = GaussNewton(),
)

# prior: pressure and temperature are not in the state, so they come from the measurement
n_air = dry_air_column(HomogeneousPath(pressure_hPa = 850.0, temperature_K = 285.0,
                                       path_length_cm = 195017.0))
x_prior = ComponentVector(H2O = 0.01 * n_air, CH4 = 1900e-9 * n_air,
                          baseline = (window_1 = zeros(5),))   # first coefficient set per measurement
prior_sigma = ComponentVector(H2O = 1.0 * n_air, CH4 = 1e-6 * n_air,
                              baseline = (window_1 = fill(1e3, 5),))

config = BatchRetrievalConfig(
    experiment = "merlin-r6",
    data_files = ["../data/20160921.h5"],
    reader = DCSReaderConfig(path_length_cm = 195017.0,
                             pressure_scale = 0.9938),   # 2016 campaign pressure correction
    noise = ResidualNoise(flat_windows_cm = [(6076.10, 6076.60), (6077.55, 6077.95)],
                          polynomial_order = 3),
    averaging_period = Minute(15),
    absorbers = absorbers,
    retrieval = retrieval,
    x_prior = x_prior,
    prior_sigma = prior_sigma,
    output_root = "../data/runs",
    n_workers = 2,           # worker processes started by run; parallel = :serial for none
)

summary = run(config)        # results in ../data/runs/<yyyy-mm-dd>_merlin-r6_<git hash>/
print(report(summary))
