# Shared fixtures for the SpectralFits test suite.
#
# The line list in data/hit20_subset.par is a small slice of HITRAN 2020
# (H2O, 12CH4, 12CO2) covering 6076-6084 and 6224-6232 cm⁻¹, so the tests
# run quickly without the full line lists.

using SpectralFits, vSmartMOM, vSmartMOM.Absorption
using Dates, LinearAlgebra, Statistics, Random, HDF5

const SF = SpectralFits
const FIXTURE_PAR = joinpath(@__DIR__, "data", "hit20_subset.par")

# spectral grids used throughout the tests
const CH4_WINDOW = (6078.0, 6082.0)
const MODEL_GRID = 6077.0:0.01:6083.0        # high-resolution cross-section grid
const MEAS_GRID = collect(6078.0:0.0067:6082.0) # instrument grid inside CH4_WINDOW
const NOISE_GRID = collect(6255.0:0.002:6255.5) # spectrally flat region used by calc_DCS_noise

const PATHLENGTH = 195017.0 # same round-trip path length read_DCS_data assigns
const P_SURF, T_SURF = 850.0, 290.0

# molecules are built once and shared between test files
const H2O_INFO = get_molecule_info("H2O", FIXTURE_PAR, 1, 1, MODEL_GRID)
const CH4_INFO = get_molecule_info("CH4", FIXTURE_PAR, 6, 1, MODEL_GRID)
const MOLECULES = [H2O_INFO, CH4_INFO]

vcd_surface(p=P_SURF, T=T_SURF) = SF.calc_vcd(p, T, PATHLENGTH)

"""Retrieval setup for a single-layer column fit with a Legendre baseline."""
function column_setup(; poly_degree=3, kwargs...)
    setup = RetrievalSetup("poly_degree" => poly_degree,
                           "fit_column" => true,
                           "linear" => false,
                           "verbose_mode" => false)
    for (k, v) in kwargs
        setup[String(k)] = v
    end
    return setup
end

"""True state for the synthetic CH4-window retrieval."""
function true_state(; p=P_SURF, T=T_SURF, H2O=0.01, CH4=1900e-9, shape=[1.0, 0.01, 0.0])
    vcd = vcd_surface(p, T)
    return StateVector("H2O" => H2O * vcd, "CH4" => CH4 * vcd,
                       "pressure" => p, "temperature" => T,
                       "shape_parameters" => shape)
end

"""Build a FrequencyCombMeasurement on `grid` with the given intensity."""
function make_measurement(intensity; grid=MEAS_GRID, p=P_SURF, T=T_SURF, σ²=1e-8,
                          time=DateTime(2020, 1, 1))
    return FrequencyCombMeasurement(intensity=collect(float.(intensity)), grid=grid,
                                    temperature=T, pressure=p, time=time,
                                    pathlength=PATHLENGTH, vcd=vcd_surface(),
                                    num_averaged_measurements=1, averaging_window=Second(1),
                                    machine_time=0.0, σ²=σ²)
end

"""Simulate a noisy single measurement from the forward model at `x_true`."""
function synthetic_measurement(x_true; σ=1e-4, rng=MersenneTwister(42), setup=column_setup())
    spectra = setup_molecules(MOLECULES)
    template = make_measurement(zeros(length(MEAS_GRID)); σ²=σ^2)
    f = generate_forward_model(x_true, template, spectra, setup)
    y = f(x_true) .+ σ .* randn(rng, length(MEAS_GRID))
    return make_measurement(y; σ²=σ^2), spectra
end

"""
Write a DCS-style HDF5 file with `n_times` spectra sampled once per minute.
Rows are simulated with the forward model on the CH4 window; the flat region
near 6255 cm⁻¹ carries baseline-only signal so calc_DCS_noise can be used.
"""
function write_synthetic_dcs(path; n_times=30, σ=1e-4, rng=MersenneTwister(7),
                             start=DateTime(2016, 9, 21, 12, 0, 30))
    grid_wn = vcat(MEAS_GRID, NOISE_GRID)
    in_window = 1:length(MEAS_GRID)
    spectra = setup_molecules(MOLECULES)
    setup = column_setup()

    p_C = fill(P_SURF / 0.9938, n_times)          # read_DCS_data rescales pressure by 0.9938
    T_C = fill(T_SURF - 273.15, n_times)          # stored in °C
    machine_time = [datetime2unix(start + Minute(k)) for k in 0:n_times-1]
    intensity = zeros(n_times, length(grid_wn))
    x_true = true_state()
    template = make_measurement(zeros(length(MEAS_GRID)))
    f = generate_forward_model(x_true, template, spectra, setup)
    y_window = f(x_true)
    for i in 1:n_times
        intensity[i, in_window] .= y_window
        intensity[i, length(MEAS_GRID)+1:end] .= x_true["shape_parameters"][1]
        intensity[i, :] .+= σ .* randn(rng, length(grid_wn))
    end

    h5open(path, "w") do fid
        fid["pressure_evolution"] = p_C
        fid["temperature_fitted_l"] = T_C
        fid["TotalTime"] = machine_time
        fid["frequency_Hz"] = grid_wn .* SF.c
        fid["WaveToFitROI_1600a"] = intensity
    end
    return (; x_true, intensity, grid_wn, machine_time)
end
