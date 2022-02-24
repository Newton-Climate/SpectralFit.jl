using Statistics, Plots, Polynomials
using SpectralFits, Dates, JLD2

# select a spectral region that has no absorption features, yet still in the middle of the DCS range
ν_min, ν_max = 6255, 6255.4
data = read_DCS_data("20160921.h5")

### take time average by co-adding spectra 
data = take_time_average(data, δt = Minute(20))

# select arbitrary measurement 
measurement =  get_measurement(2, data, ν_min, ν_max)
flat_region = measurement.intensity

# fit out the baseline with a 3rd degree polynomial
fitted = fit(measurement.grid, measurement.intensity, 3)

# evaluate the trend given the polynomial coefficients
trend = fitted.(measurement.grid)

# get the standard deviation of the detrended "measurement"
detrended = measurement.intensity - trend
σ = std(detrended)
grid = measurement.grid


# assume model of σ = a * sqrt(mean(measurement.intensity))
# we solve for a
a_averaged = σ/sqrt(mean(measurement.intensity))
@show a_averaged


p1 = plot(grid, flat_region, label="observed", color="black", ls=:dash)
plot!(grid, trend, label="trend", color="red", ls=:dot)
plot!(xlabel="wavenumber 1/cm", ylabel="intensity")

p2 = plot(grid, detrended, label="detrended obs", color="blue", ls=:dash)
plot!(xlabel="wavenumber 1/cm", ylabel="intensity")
plot(p1, p2, layout=(2,1))
plot!(fontfamily="serif-roman", legendfont=font("Computer Modern", 7))
savefig("noise.pdf")



### do a fit and calculate the χ²

include("example_run.jl")
# load a datafile so we don't have to recalculate the inverison every time 
#@load "20160921_hit16_results.JLD2" results
#out = results[2,1] # result[ co2_band, measurement_number] is the index 

# calculate the χ²
Sₑ = SpectralFits.make_obs_error(measurement, a=a_averaged);
y, f= out.measurement, out.model
χ² = ((y-f)'*Sₑ*(y-f))
χ²_r = χ²/(length(y) - length(out.x))

@show χ²_r 
