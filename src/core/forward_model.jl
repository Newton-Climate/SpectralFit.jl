# Forward models (section 3.10). The stages are generic functions; ForwardModel holds one
# object per stage, so the compiler specializes the whole chain for each combination.

"""
    ForwardModel(atmosphere, absorbers, instrument, baseline, space, units, wavenumber_fine_cm)

Callable forward model `model(x) -> Vector` for one spectral window. Built with
`build_forward_model`, which prepares the instrument and baseline, binds parameter slots and
precomputes cross sections; call the constructor directly only with prepared components.

Stages: optical depth on the fine grid (cm-1), transmission `exp(-tau)`, instrument,
measurement space, baseline.
"""
struct ForwardModel{A<:AbstractAtmosphere, S<:NamedTuple, I, B, M<:AbstractMeasurementSpace,
                    U<:AbstractStateUnits, G<:AbstractRange}
    atmosphere::A
    absorbers::S
    instrument::I
    baseline::B
    space::M
    units::U
    wavenumber_fine_cm::G
end

function (model::ForwardModel)(x::AbstractVector)
    tau = optical_depth(model.atmosphere, model.absorbers, model.units, x, model.wavenumber_fine_cm)
    signal = apply_instrument(model.instrument, model.wavenumber_fine_cm, exp.(-tau), x)
    return apply_baseline(model.baseline, model.space, to_measurement_space(model.space, signal), x)
end

n_observations(model::ForwardModel) = n_observations(model.instrument)

"""
    MultiWindowModel(windows::Tuple)

Several `ForwardModel`s sharing one state, each with its own baseline name. Calling it
concatenates the windows' outputs in order.
"""
struct MultiWindowModel{W<:Tuple}
    windows::W
end

(model::MultiWindowModel)(x::AbstractVector) = reduce(vcat, map(window -> window(x), model.windows))

n_observations(model::MultiWindowModel) = sum(n_observations, model.windows)

export ForwardModel, MultiWindowModel
