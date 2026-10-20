# Forward-model and Jacobian timings: DCS CH4 window (fixed and retrieved p, T), a layered
# profile model, and WMS with both demodulators. Run from the repository root with
#     julia --project=test benchmark/forward_model_benchmarks.jl
# Timings are the minimum over repeated calls after a warm-up call, in milliseconds.

using SpectralFits, SpectralFits.CoreLayer, ComponentArrays, Printf
import ForwardDiff

const PAR = joinpath(@__DIR__, "..", "test", "data", "hit20_subset.par")
voigt(molecule, range_cm) = SpectralFits.IOLayer.load_cross_section_model(HitranVoigt, PAR;
    molecule = molecule, isotope = 1, range_cm = range_cm)

function best_ms(f, x; repeats = 20)
    f(x)
    return minimum(1e3 * @elapsed(f(x)) for _ in 1:repeats)
end

function report_case(name, model, x; repeats = 20)
    t_model = best_ms(model, x; repeats)
    t_jac = best_ms(v -> ForwardDiff.jacobian(model, v), x; repeats = max(3, repeats ÷ 5))
    alloc = (model(x); @allocated model(x))
    @printf("%-40s model %9.3f ms   jacobian %9.3f ms   model alloc %8.1f KiB\n",
            name, t_model, t_jac, alloc / 1024)
end

fine = 6077.0:0.01:6083.0
nu = collect(6078.0:0.0067:6082.0)
atm = HomogeneousPath(pressure_hPa = 850.0, temperature_K = 290.0, path_length_cm = 195017.0)
n_air = dry_air_column(atm)
absorbers = (H2O = Absorber(:H2O, voigt(1, (6077.0, 6083.0))), CH4 = Absorber(:CH4, voigt(6, (6077.0, 6083.0))))
m = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = 1e-8)

x = ComponentVector(H2O = 0.01n_air, CH4 = 1.9e-6n_air, baseline = (window_1 = [1.0, 0.0, 0.0, 0.0, 0.0],))
report_case("DCS CH4 window, p and T fixed",
            build_forward_model(atm, absorbers, DCS(), LegendreBaseline(5), LinearSpace(),
                                ColumnDensity(), m, x; wavenumber_fine_cm = fine), x)

x_pt = ComponentVector(H2O = 0.01n_air, CH4 = 1.9e-6n_air, pressure_hPa = 850.0, temperature_K = 290.0,
                       baseline = (window_1 = [1.0, 0.0, 0.0, 0.0, 0.0],))
report_case("DCS CH4 window, p and T retrieved",
            build_forward_model(atm, absorbers, DCS(), LegendreBaseline(5), LinearSpace(),
                                ColumnDensity(), m, x_pt; wavenumber_fine_cm = fine), x_pt; repeats = 5)

p_layers = [900.0, 800.0, 700.0, 600.0, 500.0]
column = LayeredColumn(pressure_hPa = p_layers, temperature_K = [288.0, 281.0, 274.0, 266.0, 258.0],
                       solar_zenith_deg = 30.0)
n_layer = [dry_air_column(column, ComponentVector(CH4 = zeros(5)), i, VMR()) for i in 1:5] ./ column.air_mass
x_profile = ComponentVector(H2O = 0.005 .* n_layer, CH4 = 1.9e-6 .* n_layer,
                            baseline = (window_1 = [1.0, 0.0, 0.0],))
report_case("profile, 5 layers",
            build_forward_model(column, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                                ColumnDensity(), m, x_profile; wavenumber_fine_cm = fine), x_profile)

ch4 = voigt(6, (6077.0, 6083.0))
wms_atm = HomogeneousPath(pressure_hPa = 1000.0, temperature_K = 295.0, path_length_cm = 1.0e4)
times = collect(range(0.01, 0.19; length = 100))
wm = Measurement(y = zeros(length(times)), coordinate = times, noise_variance = 1.0)
tuning = LaserTuning(LinearRamp(start_cm = 6079.6, rate_cm_per_s = 4.0),
                     SinusoidalModulation(amplitude_cm = 0.1, frequency_Hz = 5e3))
laser = IntensityModulation(mean = 1.0, linear_amplitude = 0.1, linear_phase_rad = 0.0)
x_wms = ComponentVector(CH4 = 2.0e-6 * dry_air_column(wms_atm), instrument = (linear_amplitude = 0.1,))
for (name, demodulator) in (("WMS HarmonicExpansion", HarmonicExpansion()),
                            ("WMS DigitalLockIn (100 kHz)", DigitalLockIn(sampling_rate_Hz = 100e3, pre_roll_s = 0.02)))
    instrument = WavelengthModulation(tuning, laser, demodulator, HarmonicRatio(2, 1))
    model = build_forward_model(wms_atm, (CH4 = Absorber(:CH4, ch4),), instrument, NoBaseline(),
                                LinearSpace(), ColumnDensity(), wm, x_wms;
                                wavenumber_fine_cm = 6078.0:0.002:6082.5)
    report_case(name, model, x_wms; repeats = 5)
end
