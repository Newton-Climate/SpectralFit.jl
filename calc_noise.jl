using DataFrames, SingularSpectrumAnalysis, Plots, Polynomials
include("read_data.jl")
include("forward_model.jl")
include("inversion.jl")


ν_min, ν_max = 6255, 6255.4
data = read_DCS_data("../../data/DCSA/DCS_A_1/20160926.h5")
data = take_time_average(data)
measurement =  get_measurement(1, data, ν_min, ν_max)

data = DataFrame((X=measurement.grid, Y=measurement.intensity))
#ols = lm(@formula(Y ~ X), data)
#@show ols
L = Int(floor(length(measurement.intensity)/2))
#trend, season = analyze(measurement.intensity, L)
k = hcat(measurement.grid.^2, measurement.grid, ones(size(measurement.grid)))
     y = measurement.intensity
     ols = inv(k'*k)*k'*y
m, b = ols[end-1], ols[end]
a = ols[1]
trend = a*measurement.grid.^2 + m*measurement.grid + b*ones(size(measurement.grid))
fitted = fit(measurement.grid, measurement.intensity, 2)
trend = fitted.(measurement.intensity)
detrended = measurement.intensity - trend
σ = std(detrended)
grid = measurement.grid

p1 = plot(grid, measurement.intensity, label="observed", color="black", ls=:dash)
plot!(grid, trend, label="trend", color="red", ls=:dot)

plot!(xlabel="cm⁻¹", ylabel="intensity")

p2 = plot(grid, detrended, label="detrended obs", color="blue", ls=:dash)
plot!(xlabel="cm⁻¹", ylabel="intensity")
plot(p1, p2, layout=(2,1))
savefig("noise.pdf")

