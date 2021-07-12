using DataFrames, SingularSpectrumAnalysis, Plots, Polynomials
include("read_data.jl")
include("forward_model.jl")
include("inversion.jl")


ν_min, ν_max = 6255, 6255.4
data = read_DCS_data("../../data/DCSA/DCS_A_1/20160926.h5")

#
#
measurement =  get_measurement(30, data, ν_min, ν_max)




#L = Int(floor(length(measurement.intensity)/2))

#k = hcat(measurement.grid.^2, measurement.grid, ones(size(measurement.grid)))

fitted = fit(measurement.grid, measurement.intensity, 3)
trend = fitted.(measurement.grid)
detrended = measurement.intensity - trend
σ = std(detrended)
grid = measurement.grid

a = σ/sqrt(mean(measurement.intensity))
m1 = σ

### Time averaged
data = take_time_average(data)
measurement =  get_measurement(2, data, ν_min, ν_max)

fitted = fit(measurement.grid, measurement.intensity, 3)
trend = fitted.(measurement.grid)
detrended = measurement.intensity - trend
σ = std(detrended)
grid = measurement.grid
m2 = σ

a_averaged = σ/sqrt(mean(measurement.intensity))

@show a
@show a_averaged

p1 = plot(grid, measurement.intensity, label="observed", color="black", ls=:dash)
plot!(grid, trend, label="trend", color="red", ls=:dot)
plot!(xlabel="wavenumber 1/cm", ylabel="intensity")

p2 = plot(grid, detrended, label="detrended obs", color="blue", ls=:dash)
plot!(xlabel="wavenumber 1/cm", ylabel="intensity")
plot(p1, p2, layout=(2,1))
plot!(fontfamily="serif-roman", legendfont=font("Computer Modern", 7))
savefig("noise.pdf")

