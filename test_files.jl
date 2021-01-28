include("read_data.jl")
include("forward_model.jl")
include("inversion.jl")
using JLD, Statistics 


inversion_setup = Dict{String,Real}(
    "poly_degree" => 40,
    "fit_pT" => true,
    "use_OCO" => false,
    "use_TCCON" => false,
    "take_time_average" => true
)

#data = read_NIST_data("../../data/20160921.h5")
data = read_new_NIST_data("../../data/20161019.h5");
data = take_time_average(data)
measurement =  get_measurement(10, data, 6050, 6200)

#xₐ = [0.01; 2000.0e-9; 400.0e-6; 0.001; mean(data.temperature); mean(data.pressure); maximum(data.intensity); zeros(inversion_setup["poly_degree"]-1)];
xₐ = [0.01; 2000.0e-9; 400.0e-6; 0.001; mean(data.temperature); mean(data.pressure); maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)];

invert_all_files(xₐ, inversion_setup, path="/Users/newtn/projects/FreqComb/data/DCS_A/DCS_A_1/")
#save("co2.jld", "con", out)
