using RadiativeTransfer, OrderedCollections
using SpectralFits
# define the reetrieval parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 20,
    "fit_pressure" => true,
    "fit_temperature" => true,
    "use_OCO" => true,
"use_TCCON" => false)

# Just defining the spectral windows for each species
ν_CH4 = (6055, 6120)
ν_CO2 = (6205, 6255);
ν_HDO = (6310,6380);

# Read the DCS DAta 
data = read_DCS_data("../../data/DCSA/DCS_A_1/20160926.h5")
data = take_time_average(data)
measurement =  get_measurement(1, data, ν_CO2[1], ν_CO2[2]) # get 1 measurement 

ν_min, ν_max = ν_CO2[1]-1, ν_CO2[2]+1
ν_range = ν_min:ν_max

# Get the HiTran parameters
CH₄ = get_molecule_info("../CH4_S.data", 6, 1, ν_range)
#_¹³CH₄ = get_molecule_info("../13CH4_S.data", 6, 2, ν_range)
#H₂O = get_molecule_info("../../data/linelists/2020_experimental/atm.161", 1, 1, ν_range)
H₂O = get_molecule_info("../H2O_S.data", 1, 1, ν_range)
CO₂ = get_molecule_info("../CO2_S.data", 2,1,ν_range)
HDO = get_molecule_info("../HDO_S.data", 1,4,ν_range)

# Calculate the cross-sections and store in dictionary
molecules = [H₂O, CH₄, CO₂, HDO]
spec = construct_spectra(molecules, ν_min=ν_min, δν=0.01, ν_max=ν_max, p=measurement.pressure, T=measurement.temperature)

# spectral windows for fitting
# key = spectral window
# value = what we are retrieving in that window
spectral_windows = OrderedDict(ν_CH4 => (CH₄,),
                        ν_CO2 => (CO₂, H₂O, "temperature", "pressure"))

# define the initial guess 
xₐ = OrderedDict{Any,Any}(H₂O => 0.01,
    CH₄ => 2000e-9,
                  CO₂ => 400e-6,
                  HDO => 0.0001,
                  "pressure" => measurement.pressure,
                  "temperature" => measurement.temperature,
                  "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])

# just testing the fit itself
out = nonlinear_inversion(xₐ, measurement, spec, inversion_setup)

# test the full code
#run_inversion(xₐ, data, molecules, inversion_setup, spectral_windows)
@show out.χ²
