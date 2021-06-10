using RadiativeTransfer, OrderedCollections, Plots 

include("read_data.jl")
include("forward_model.jl")
include("inversion.jl")
include("profile.jl")
using Plots


num_layers = 20
# Just defining the spectral windows for each species
ν_CH4 = (6055, 6120)
                        ν_CO2 = (6180, 6260);
ν_HDO = (6310,6380);

ν_min, ν_max = ν_CO2[1], ν_CO2[2]
# Read the DCS DAta 
data = read_DCS_data("../../data/DCSA/DCS_A_1/20160926.h5")
data = take_time_average(data)
measurement =  get_measurement(1, data, ν_min, ν_max) 


ν_range = 6000:6300

# Get the HiTran parameters
CH₄ = get_molecule_info("../CH4_S.data", 6, 1, ν_range)
#_¹³CH₄ = get_molecule_info("../13CH4_S.data", 6, 2, ν_range)
#H₂O = get_molecule_info("../../data/linelists/2020_experimental/atm.161", 1, 1, ν_range)
H₂O = get_molecule_info("../H2O_S.data", 1, 1, ν_range)
CO₂ = get_molecule_info("../CO2_S.data", 2,1,ν_range)
#HDO = get_molecule_info("../HDO_S.data", 1,4,ν_range)

# Calculate the cross-sections and store in dictionary
molecules = [H₂O, CO₂, CH₄]
p = collect(range(440, measurement.pressure, length=num_layers))
T = collect(range(250,measurement.temperature, length=num_layers))
spec = construct_spectra(molecules, ν_min=ν_min-1, δν=0.003, ν_max=ν_max+1, p, T)

# spectral windows for fitting
# key = spectral window
# value = what we are retrieving in that window
spectral_windows = OrderedDict(ν_CH4 => (CH₄,),
                        ν_CO2 => (CO₂, H₂O, "temperature", "pressure"))
a = ones(size(p))


# define the reetrieval parameters
inversion_setup = Dict{String,Any}(
    "poly_degree" => 10,
    "fit_pressure" => true,
    "fit_temperature" => true,
    "use_OCO" => true,
"use_TCCON" => false)


# define the initial guess 
x_true = OrderedDict{Any,Any}(H₂O => collect(range(0.01, 0.03, length=num_layers)),
                              CO₂ => 1e-6 * collect(range(395, 400, length=num_layers)),
                              CH₄ => 1e-9*collect(range(1800, 2000, length=num_layers)),
                  "pressure" => p,
                  "temperature" => T,
                              "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])


# Make some synthetic data 
f = generate_profile_model(x_true, measurement, spec, inversion_setup);
τ = f(x_true)

# noise 
ϵ = 0.005092707186368767*sqrt(mean(τ))* randn.(length(τ))
measurement.intensity = τ #+ ϵ



xₐ = OrderedDict{Any,Any}(H₂O => 0.02*a,
                          CO₂ => 395e-6*a,
                          CH₄ => 1800e-9*a,
                          "shape_parameters" => [maximum(measurement.intensity); zeros(inversion_setup["poly_degree"]-1)])
inversion_setup["fit_pressure"] = false

σ = OrderedDict{Any,Any}(H₂O => 0.01*a,
                         CO₂ => 4.0e-6*a,
                         CH₄ => 200e-9*a,
                         "shape_parameters" => ones(inversion_setup["poly_degree"]))
#1e-10*ones(inversion_setup["poly_degree"])) 
inversion_setup = push!(inversion_setup, "σ" => σ)



println("beginning fit")
f = generate_profile_model(xₐ, measurement, spec, inversion_setup)
out = nonlinear_inversion(f, xₐ, measurement, spec, inversion_setup)
#@show out.x[11:20]*1e6
@show out.χ²

x_retrieved = assemble_state_vector!(out.x, collect(keys(xₐ)), num_layers, inversion_setup)



x_retrieved[CO₂] = x_retrieved[CO₂] ./ (1 .- x_retrieved[H₂O])
co2_con = 1e6*x_retrieved[CO₂]
@show co2_con
h2o_con = 1e2*x_retrieved[H₂O]
### Plot our results
p1 = plot(measurement.grid, measurement.intensity, label="observed", color="black")
plot!(measurement.grid, out.y, label="modelled", color="red")
plot!(xlabel="wave-number", ylabel="intensity")

p2=plot(measurement.grid, measurement.intensity - out.y, label="observed - modelled")
plot(p1, p2, layout=(2,1))
savefig("profile_CO2_fit.pdf")

### plot the profile
p3 = plot(1e6*xₐ[CO₂], p, label="a priori", yaxis=:flip, lw=2, color="green")
plot!(1e6*x_true[CO₂], p, label="truth", yaxis=:flip, lw=2, color="black")
plot!(1e6*x_retrieved[CO₂], p, label="retrieved", lw=2, yaxis=:flip, color="red")
plot!(xlabel="ppm CO₂", ylabel="mbar")



ν_min, ν_max = ν_CH4[1], ν_CH4[2]
measurement =  get_measurement(1, data, ν_min, ν_max)
spec = construct_spectra(molecules, ν_min=ν_min-1, δν=0.003, ν_max=ν_max+1, p, T)
# Make some synthetic data
f = generate_profile_model(x_true, measurement, spec, inversion_setup);
τ = f(x_true)

# noise 
ϵ = 0.005092707186368767*sqrt(mean(τ))* randn.(length(τ))
measurement.intensity = τ #+ ϵ
inversion_setup["fit_pressure"] = false
println("beginning fit")
f = generate_profile_model(xₐ, measurement, spec, inversion_setup)
out2 = nonlinear_inversion(f, xₐ, measurement, spec, inversion_setup)

x_retrieved = assemble_state_vector!(out2.x, collect(keys(xₐ)), num_layers, inversion_setup)


x_retrieved[CH₄] = x_retrieved[CH₄] ./ (1 .- x_retrieved[H₂O])

ch4_con = x_retrieved[CH₄]
@show 1e9*x_retrieved[CH₄]

# plot water 
p4 = plot(1e2*xₐ[H₂O], p, label="a priori", yaxis=:flip, lw=2, color="green")
plot!(1e2*x_true[H₂O], p, label="truth", yaxis=:flip, lw=2, color="black")
plot!(1e2*x_retrieved[H₂O], p, label="retrieved", lw=2, yaxis=:flip, color="red")
plot!(xlabel="percent H₂O", ylabel="mbar")


p5 = plot(1e9*xₐ[CH₄], p, label="a priori", yaxis=:flip, lw=2, color="green")
plot!(1e9*x_true[CH₄], p, label="truth", yaxis=:flip, lw=2, color="black")
plot!(1e9*x_retrieved[CH₄], p, label="retrieved", lw=2, yaxis=:flip, color="red")
plot!(xlabel="ppb CH₄", ylabel="mbar")
plot(p3, p4, p5, layout=(1,3))
savefig("vertical_profile.pdf")
#savefig("profile_CO2.pdf")

### Plot our results
p6 = plot(measurement.grid, measurement.intensity, label="observed", color="black")
plot!(measurement.grid, out2.y, label="modelled", color="red")
plot!(xlabel="wave-number", ylabel="intensity")

p7=plot(measurement.grid, measurement.intensity - out2.y, label="observed - modelled")
plot(p6, p7, layout=(2,1))
savefig("profile_CH4_fit.pdf")


### averaging kernals and error analysis
A1 = out.G*out.K
A2 = out2.G*out2.K
vcd = make_vcd_profile(p, T, x_retrieved[H₂O])
h2o_ind = 1:num_layers
co2_ind = num_layers+1:2*num_layers
ch4_ind = 2*num_layers+1:3*num_layers

H = vcd ./ sum(vcd)
cK_h2o = (H'*A2[h2o_ind, h2o_ind])'
cK_co2 = (H'*A1[co2_ind, co2_ind])'
cK_ch4 = (H'*A2[ch4_ind, ch4_ind])'

p_co2_ck = plot(cK_co2, p, yflip=true,lw=2, label="cAK for CO2")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for CO2")

p_ch4_ck = plot(cK_ch4, p, yflip=true,lw=2, label="cAK for CH4")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for CH4")

p_h2o_ck = plot(cK_h2o, p, yflip=true,lw=2, label="cAK for H2O")
plot!(xlabel="Column averaging kernel", ylabel="Pressure [hPa]", title="Column averaging kernel for H2O")
plot(p_h2o_ck, p_co2_ck, p_ch4_ck, layout=(1,3))
savefig("column_averaging_kernal.pdf")
