using RadiativeTransfer, OrderedCollections, Dates



using Plots, SpectralFits

# define the reetrieval parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 100,
    "fit_pressure" => true,
    "fit_temperature" => true,
    "use_OCO" => false,
"use_TCCON" => false,
"verbose_mode" => true,
"architecture" => CPU())

# Just defining the spectral windows for each species
ν_CH4 = (6055, 6120)
                        ν_CO2 = (6205, 6255);
ν_HDO = (6310,6380);

# Read the DCS DAta 
data = read_DCS_data("20160921.h5")
data = take_time_average(data, δt=Minute(20))
measurement =  get_measurement(1, data, ν_CO2[1], ν_CO2[2]) # get 1 measurement 

ν_min, ν_max = ν_CO2[1]-1, ν_CO2[2]+1
ν_range = ν_min:ν_max

# Get the HiTran parameters
CH₄ = get_molecule_info("CH4", "../CH4_S.data", 6, 1, ν_range, architecture=inversion_setup["architecture"])
H₂O = get_molecule_info("H2O", "../H2O_S.data", 1, 1, ν_range, architecture=inversion_setup["architecture"])
CO₂ = get_molecule_info("CO2", "../CO2_S.data", 2,1,ν_range, architecture=inversion_setup["architecture"])

# Calculate the cross-sections and store in dictionary
molecules = [H₂O, CH₄, CO₂]
spec = construct_spectra(molecules, ν_grid=ν_min:0.003:ν_max, p=measurement.pressure, T=measurement.temperature)

# spectral windows for fitting
# key = spectral window
# value = what we are retrieving in that window
spectral_windows = OrderedDict(ν_CH4 => (CH₄,),
                        ν_CO2 => (CO₂, H₂O, "temperature", "pressure"))

# define the initial guess 
xₐ = OrderedDict{Any,Any}("H2O" => 0.01,
    "CH4" => 2000e-9,
                  "CO2" => 400e-6,
                  "pressure" => measurement.pressure,
                  "temperature" => measurement.temperature,
                  "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])

# just testing the fit itself
f = generate_forward_model(xₐ, measurement, spec, inversion_setup);
tau = f(assemble_state_vector!(xₐ))
out = nonlinear_inversion(f, xₐ, measurement, spec, inversion_setup)
