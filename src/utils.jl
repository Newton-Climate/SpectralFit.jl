
function save_inversion_results(filename::String, results::Array{InversionResults}, data::Dataset, experiment_label::Array{String})
    
    file = NCDataset(filename, "c");
    num_experiments, num_datapoints = size(results);

    # create the dimensions
    #num_datapoints = 2
           defDim(file, "start_time", num_datapoints)
    defVar(file, "start_time", data.timestamp, ("start_time",))

    # save the concentrations
    H₂O_VMR = defVar(file, "H2O_VMR", Float64, ("start_time",))
    H₂O_VMR[:] = [results[1,i].x[H₂O_ind] for i =1:num_datapoints]
    
    CO₂_VMR = defVar(file, "CO2_VMR", Float64, ("start_time",))
    CO₂_VMR[:] = [results[1,i].x[CO₂_ind] for i=1:num_datapoints]
    CH₄_VMR = defVar(file, "CH4_VMR", Float64, ("start_time",))
    CH₄_VMR[:] = [results[CH₄_ind,i].x[CH₄_ind] for i=1:num_datapoints];
    HDO_VMR = defVar(file, "HDO_VMR", Float64, ("start_time",))
    HDO_VMR[:] = [results[3,i].x[HDO_ind] for i=1:num_datapoints];

    pressure = defVar(file, "pressure", Float64, ("start_time",))
    pressure[:] = [results[1,i].x[pressure_ind] for i=1:num_datapoints];

    temperature = defVar(file, "temperature", Float64, ("start_time",))
    temperature[:] = [results[1,i].x[temperature_ind] for i=1:num_datapoints];
    try

    

        for i=1:num_experiments
            println(experiment_label[i])
       timeseries = results[i,:]
           
       group = defGroup(file, experiment_label[i])
        defDim(group, "spectral_grid", length(timeseries[1].grid))
        defVar(group, "spectral_grid", timeseries[1].grid, ("spectral_grid",))
        model = defVar(group, "model", Float64, ("start_time", "spectral_grid"))
        measurement = defVar(group, "measurement", Float64, ("start_time", "spectral_grid"))

            # save to file
            println("saving measurement")
            measurement[:,:] = list2array([timeseries[i].y for i=1:num_datapoints])
            println("saving model")
        model[:,:] = list2array([timeseries[i].f for i=1:num_datapoints])
    end
    catch e
        println("save failed")
        println(e)
    finally
        close(file)
    end
end


"""
Finds the indexes given values ν_min:ν_max
"""

function find_indexes(ν_min::Real, ν_max::Real, ν_grid::Array{Float64,1})
    
    a = findlast(x -> x <= ν_min, ν_grid)
    b = findfirst(x -> x >= ν_max, ν_grid)
    indexes = collect(a:b)
    return indexes
end #function find_indexes


function calc_vcd(p::Float64, T::Float64, δz::Float64, VMR_H₂O::Float64)
    """
Calculates the vertical column density
"""
    
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end # function calc_vcd

function calc_vcd(p::Float64, T::Float64, δz::Float64)
    VMR_H₂O = 0
    ρₙ = p*(1-VMR_H₂O) / (r*T)*Nₐ/1.0e4
    vcd = δz*ρₙ
    return vcd
end #function calc_vcd


function assemble_state_vector!(x::AbstractDict)
    out::Array{Real,1} = []
    for key in keys(x)
        out = append!(out, x[key])
        end
    return out
end #function assemble_state_vector!

function assemble_state_vector!(x::Vector{<:Real}, key_vector::Array{Any,1}, inversion_setup::AbstractDict)
    out::OrderedDict{Any,Any} = OrderedDict([key_vector[i] => x[i] for i=1:length(key_vector)-1])
    out = push!(out, "shape_parameters" => x[end-inversion_setup["poly_degree"]+1:end])
    return out
end #function assemble_state_vector!


"""
calculates the legendre polynomial over domain x::Vector of degree max::Integer
"""
function compute_legendre_poly(x::Array{<:Real,1}, nmax::Integer)
    
    FT = eltype(x)
    @assert nmax > 1
    #@assert size(P) == (nmax,length(x))
    P⁰ = zeros(nmax,length(x));
   
    # 0th Legendre polynomial, a constant
    P⁰[1,:] .= 1;

    # 1st Legendre polynomial, x
    P⁰[2,:] = x;

    for n=2:nmax-1
        for i in eachindex(x)
            l = n-1
            P⁰[n+1,i] = ((2l + 1) * x[i] * P⁰[n,i] - l * P⁰[n-1,i])/(l+1)
        end
    end
    return P⁰
end  
