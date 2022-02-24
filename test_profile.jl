using RadiativeTransfer, OrderedCollections, Plots, LaTeXStrings
using Statistics, LinearAlgebra
using SpectralFits, JLD2

snr = 1000.0;

# define the reetrieval parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 50,
    "fit_pressure" => false,
    "fit_temperature" => false,
    "use_OCO" => false,
    "use_TCCON" => false,
"verbose_mode" => true)
num_windows = 4
num_layers = 40
# Just defining the spectral windows for each species
ν_CH4 = (6055, 6120)
                        ν_CO2 = (6180, 6255);
ν_HDO = (6310,6380);

ν_min, ν_max = 6050, 6260
# Read the DCS DAta 
data = read_DCS_data("20160921.h5")
#data = take_time_average(data)

p = collect(range(250, 850, length=num_layers))
T = collect(range(100, 285, length=num_layers))
measurement =  get_measurement(1, data, ν_CO2[1], ν_CO2[2], p, T) 
ν_range = 6000:6400

# Get the HiTran parameters
CH₄ = get_molecule_info("CH4", "../CH4_S.data", 6, 1, ν_range)
H₂O = get_molecule_info("H2O", "../H2O_S.data", 1, 1, ν_range)
CO₂ = get_molecule_info("CO2", "../CO2_S.data", 2,1,ν_range)

# Calculate the cross-sections and store in dictionary
molecules = [H₂O, CH₄, CO₂]
println("first spectra calculation")
spec = construct_spectra_by_layer(molecules, ν_min=ν_CO2[1]-1, δν=0.003, ν_max=ν_CO2[2]+1, p=p, T=T)
a = ones(size(p))
println("done with spectra calculation")


# define the initial guess 
x_true = OrderedDict{Any,Any}("H2O" => collect(range(0.01, 0.03, length=num_layers)),
                              "CO2" => 1e-6 * collect(range(395, 400, length=num_layers)),
                              "CH4" => 1e-9*collect(range(1800, 2000, length=num_layers)),
                  "pressure" => p,
                  "temperature" => T,
                              "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])

# define the initial guess 
xₐ = OrderedDict{Any,Any}("H2O" => 0.01*a,
    "CH4" => 2000e-9*a,
                  "CO2" => 397e-6*a,
                  "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])

# just testing the fit itself
f = generate_profile_model(x_true, measurement, spec, inversion_setup);

τ = f(x_true)
signal = mean(τ)
σ = signal / snr


# noise 
ϵ = randn(length(τ)) * σ
measurement.intensity = τ + ϵ
measurement.σ = σ
@show σ



# define prior uncertainty 
σ = OrderedDict{Any,Any}("H2O" => 0.01*a,
                                                  "CH4" => 200e-9*a,
                         "CO2" => 4.0e-6*a,
#                         "pressure" => ones(num_layers),
#                         "temperature" => ones(num_layers),
                         "shape_parameters" => ones(inversion_setup["poly_degree"]))
inversion_setup = push!(inversion_setup, "σ" => σ)


println("beginning CO2 fit")
inversion_setup["fit_temperature"], inversion_setup["fit_pressure"] = false, false 
f = generate_profile_model(xₐ, measurement, spec, inversion_setup);
out = nonlinear_inversion(f, xₐ, measurement, spec, inversion_setup)
@show out.χ²

# Make some synthetic data 


#save("co2_retrieval.jld", "K",out.K, "Sa",out.Sₐ, "Se",out.Sₑ, "measurement", out.measurement, "model",out.model, "grid", out.grid)

# convert output to a dict
x_retrieved = assemble_state_vector!(out.x, collect(keys(xₐ)), num_layers, inversion_setup)

co2_con = 1e6*x_retrieved["CO2"]
@show co2_con
h2o_con = 1e2*x_retrieved["H2O"]


### Plot our results
p1 = plot(measurement.grid, measurement.intensity, label="observed", color="black")
plot!(measurement.grid, out.model, label="modelled", color="red")
plot!(xlabel="wave-number", ylabel="intensity")

p2=plot(measurement.grid, measurement.intensity - out.model, label="observed - modelled")
plot(p1, p2, layout=(2,1))
savefig("profile_CO2_fit.pdf")

### plot the profile
p3 = plot(1e6*xₐ["CO2"], p, yaxis=:flip, lw=2, color="green")
plot!(1e6*x_true["CO2"], p, yaxis=:flip, lw=2, color="black")
plot!(1e6*x_retrieved["CO2"], p, lw=2, yaxis=:flip, color="red")
plot!(xlabel=L"\textrm{[CO}_2\textrm{] ppm}", ylabel="mbar", legend=false)



# extract measurement in CH4 range 
ν_min, ν_max = ν_CH4[1], ν_CH4[2]
measurement =  get_measurement(1, data, ν_min, ν_max, p, T)
spec = construct_spectra_by_layer(molecules, ν_min=ν_min-1, δν=0.003, ν_max=ν_max+1, p=p, T=T)

# Make some synthetic data
f = generate_profile_model(x_true, measurement, spec, inversion_setup);
τ = f(x_true)
signal = mean(τ)
σ = signal / snr


# noise
ϵ = randn(length(τ)) * σ
measurement.intensity = τ + ϵ
measurement.σ = σ
println("beginning fit over CH4 range")
inversion_setup["fit_temperature"], inversion_setup["fit_pressure"] = false, false 
f = generate_profile_model(xₐ, measurement, spec, inversion_setup);
out2 = nonlinear_inversion(f, xₐ, measurement, spec, inversion_setup)
#save("ch4_retrieval.jld", "K", out2.K, "Sa", out2.Sₐ, "Se",out2.Sₑ, "measurement", out2.measurement, "model",out2.model, "grid", out2.grid)

# convert to Dict 
x_retrieved = assemble_state_vector!(out2.x, collect(keys(xₐ)), num_layers, inversion_setup)



ch4_con = x_retrieved["CH4"]
@show 1e9*x_retrieved["CH4"]

# plot water 
p4 = plot(1e2*xₐ["H2O"], p, yaxis=:flip, lw=2, color="green")
plot!(1e2*x_true["H2O"], p, yaxis=:flip, lw=2, color="black")
plot!(1e2*x_retrieved["H2O"], p, lw=2, yaxis=:flip, color="red")
plot!(xlabel=L"\textrm{[H}_2\textrm{O] percent}", ylabel="mbar", legend=false)


p5 = plot(1e9*xₐ["CH4"], p, label="a priori", yaxis=:flip, lw=2, color="green")
plot!(1e9*x_true["CH4"], p, label="truth", yaxis=:flip, lw=2, color="black")
plot!(1e9*x_retrieved["CH4"], p, label="retrieved", lw=2, yaxis=:flip, color="red")
plot!(xlabel=L"\textrm{[CH}_4\textrm{] ppb}", ylabel="mbar")
plot(p3, p4, p5, layout=(1,3))
plot!(fontfamily="serif-roman", legendfont=font("Computer Modern", 7))
savefig("vertical_profile_with_noise.pdf")
#savefig("profile_CO2.pdf")

### Plot our results
p6 = plot(measurement.grid, measurement.intensity, label="observed", color="black")
plot!(measurement.grid, out2.model, label="modelled", color="red")
plot!(xlabel="wave-number", ylabel="intensity")

p7=plot(measurement.grid, measurement.intensity - out2.model, label="observed - modelled")
plot(p6, p7, layout=(2,1))
savefig("profile_CH4_fit.pdf")



### averaging kernals and error analysis
# calculate averaging kernal

#include("calc_noise-CF.jl")

#out.Sₑ = Diagonal(1/σ.^2[1] * ones(length(out.grid)))
#out2.Sₑ = Diagonal(1/σ.^2[1] * ones(length(out2.grid)))

# modify the Se matrix a bit (remove later)
G1 = calc_gain_matrix(out)
G2 = calc_gain_matrix(out2)
A1 = G1*out.K
A2 = G2*out2.K
vcd = make_vcd_profile(p, T)

h2o_ind = 1:num_layers
ch4_ind = num_layers+1:2*num_layers
co2_ind = 2*num_layers+1:3*num_layers

# weighting function 
H = vcd ./ sum(vcd)

co2_degrees = tr(A1[co2_ind,co2_ind])
h2o_degrees = tr(A1[h2o_ind, h2o_ind])
ch4_degrees = tr(A2[ch4_ind, ch4_ind])

@show co2_degrees
@show h2o_degrees
@show ch4_degrees

# column weighted averaging kernal 
cK_h2o = (H'*A2[h2o_ind, h2o_ind])' ./ H
cK_co2 = (H'*A1[co2_ind, co2_ind])' ./ H
cK_ch4 = (H'*A2[ch4_ind, ch4_ind])' ./ H

### plot averaging kernals 
p_co2_ck = plot(cK_co2, p, yflip=true,lw=2, label="cAK for CO2")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for CO2")

p_ch4_ck = plot(cK_ch4, p, yflip=true,lw=2, label="cAK for CH4")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for CH4")

p_h2o_ck = plot(cK_h2o, p, yflip=true,lw=2, label="cAK for H2O")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for H2O")
plot(p_h2o_ck, p_co2_ck, p_ch4_ck, layout=(1,3))
savefig("column_averaging_kernal.pdf")

co2_results, ch4_results = out, out2

@show out.χ²
@show out2.χ²

@save "vertical_provile.JLD2" co2_ind ch4_ind h2o_ind co2_results ch4_results A1 A2 cK_co2 cK_ch4 cK_h2o p T
