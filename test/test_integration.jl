# Integration tests: forward models and retrievals on synthetic spectra
# simulated with the package's own forward model (an identical-twin OSSE).

using ForwardDiff, JLD2

@testset "single-layer forward model" begin
    x = true_state()
    meas = make_measurement(zeros(length(MEAS_GRID)))
    spectra = setup_molecules(MOLECULES)
    f = generate_forward_model(x, meas, spectra, column_setup())

    y = f(x)
    @test length(y) == length(MEAS_GRID)
    @test all(isfinite, y)
    baseline = vec(calc_polynomial_term(3, x["shape_parameters"], length(MEAS_GRID)))
    @test all(y .<= baseline .+ 1e-12)          # absorption only removes light
    @test minimum(y ./ baseline) < 0.99         # and the CH4/H2O lines are visible

    # dictionary and vector state vectors give the same spectrum
    v = assemble_state_vector!(x)
    @test f(v) ≈ y

    # ForwardDiff Jacobian agrees with a central finite difference
    K = ForwardDiff.jacobian(f, v)
    @test size(K) == (length(MEAS_GRID), length(v))
    for j in (1, 2, 4)                          # H2O, CH4, temperature
        h = 1e-4 * abs(v[j])
        e = zeros(length(v)); e[j] = h
        fd = (f(v .+ e) .- f(v .- e)) ./ (2h)
        @test K[:, j] ≈ fd rtol = 1e-4 atol = 1e-6 * maximum(abs, fd)
    end

    # with no absorbers the model returns the Legendre baseline
    empty = copy(x); empty["H2O"] = 0.0; empty["CH4"] = 0.0
    @test f(empty) ≈ baseline
end

"""State without p and T: the forward model then uses the measured p and T."""
gas_state(x) = OrderedDict(k => v for (k, v) in x if k ∉ ("pressure", "temperature"))

@testset "nonlinear_inversion recovers p, T and gases (noise-free)" begin
    x_true = true_state()
    meas, spectra = synthetic_measurement(x_true; σ=0.0)
    meas.σ² = 1e-8
    vcd = vcd_surface()
    xa = StateVector("H2O" => 0.012 * vcd, "CH4" => 1700e-9 * vcd,
                     "pressure" => 800.0, "temperature" => 285.0,
                     "shape_parameters" => [maximum(meas.intensity), 0.0, 0.0])
    setup = column_setup()
    f = generate_forward_model(xa, meas, spectra, setup)
    res = nonlinear_inversion(f, xa, meas, spectra, setup)
    for key in keys(x_true)
        @test res.x[key] ≈ x_true[key] rtol = 1e-4 atol = 1e-6
    end
end

@testset "nonlinear_inversion recovers the true state" begin
    σ = 1e-5
    x_true = gas_state(true_state())
    meas, spectra = synthetic_measurement(x_true; σ=σ)
    setup = column_setup()
    vcd = vcd_surface()
    xa = StateVector("H2O" => 0.012 * vcd, "CH4" => 1700e-9 * vcd,
                     "shape_parameters" => [maximum(meas.intensity), 0.0, 0.0])
    f = generate_forward_model(xa, meas, spectra, setup)
    res = nonlinear_inversion(f, xa, meas, spectra, setup)

    @test res isa InversionResults
    @test collect(keys(res.x)) == collect(keys(x_true))
    S = res.S
    @test isposdef(Symmetric(S))
    @test 0.7 < res.χ² < 1.3                  # residuals consistent with the noise
    for (i, key) in enumerate(["H2O", "CH4"])
        @test abs(res.x[key] - x_true[key]) < 4 * sqrt(S[i, i])
    end
    @test sqrt(S[2, 2]) / x_true["CH4"] < 0.02  # CH4 is well constrained at this SNR
    @test res.x["CH4"] / vcd ≈ 1900e-9 rtol = 0.05
    @test res.model ≈ f(res.x)
end

@testset "nonlinear_inversion with masked windows" begin
    x_true = gas_state(true_state())
    meas, spectra = synthetic_measurement(x_true; σ=1e-5)
    setup = column_setup(masked_windows=[6079.0 6079.2])
    f = generate_forward_model(x_true, meas, spectra, setup)
    res = nonlinear_inversion(f, x_true, meas, spectra, setup)
    masked = SF.find_mask(meas.grid, [6079.0 6079.2])
    @test !isempty(masked)
    @test all(diag(res.Sₑ⁻¹)[masked] .< 1e-3 * maximum(diag(res.Sₑ⁻¹)))
    @test res.x["CH4"] ≈ x_true["CH4"] rtol = 0.05
end

"""Two-layer profile measurement and state used by the profile retrievals."""
function profile_problem(; σ=1e-4, rng=MersenneTwister(11))
    p, T = [850.0, 500.0], [290.0, 260.0]
    vcd = make_vcd_profile(p, T)
    x_true = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => 0.005 .* vcd, "CH4" => [1950e-9, 1850e-9] .* vcd,
        "shape_parameters" => [1.0, 0.01, 0.0])
    template = make_measurement(zeros(length(MEAS_GRID)); p=p, T=T, σ²=σ^2)
    template.vcd = vcd
    setup = column_setup()
    spectra = setup_molecules(MOLECULES)
    f = generate_profile_model(x_true, template, spectra, setup)
    y = f(assemble_state_vector!(x_true)) .+ σ .* randn(rng, length(MEAS_GRID))
    meas = make_measurement(y; p=p, T=T, σ²=σ^2)
    meas.vcd = vcd

    xa = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => 0.006 .* vcd, "CH4" => [1800e-9, 1800e-9] .* vcd,
        "shape_parameters" => [maximum(y), 0.0, 0.0])
    setup["σ"] = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => 0.005 .* vcd, "CH4" => [300e-9, 300e-9] .* vcd,
        "shape_parameters" => [1.0, 1.0, 1.0])
    setup["γ"] = 0.0
    return (; x_true, xa, meas, spectra, setup, vcd)
end

@testset "profile retrieval" for inversion in (profile_inversion, adaptive_inversion)
    prob = profile_problem()
    f = generate_profile_model(prob.xa, prob.meas, prob.spectra, prob.setup)
    res = inversion(f, prob.xa, prob.meas, prob.spectra, prob.setup)

    @test res isa InversionResults
    @test length(res.x["CH4"]) == 2
    @test all(isfinite, assemble_state_vector!(res.x))
    @test 0.5 < res.χ² < 1.5
    # the total CH4 column is well constrained even if the split between layers is not
    @test sum(res.x["CH4"]) ≈ sum(prob.x_true["CH4"]) rtol = 0.01
    # the posterior is closer to the truth than the prior
    @test abs(sum(res.x["CH4"]) - sum(prob.x_true["CH4"])) <
          abs(sum(prob.xa["CH4"]) - sum(prob.x_true["CH4"]))
end

@testset "profile_inversion stops once the fit converges" begin
    prob = profile_problem()
    f = generate_profile_model(prob.xa, prob.meas, prob.spectra, prob.setup)
    y = prob.meas.intensity
    # each iteration evaluates the model once with plain Float64 input (after the step);
    # record the residual norm of each of those evaluations
    r = [norm(f(assemble_state_vector!(prob.xa)) - y)]
    logged(x) = (fx = f(x); eltype(x) <: AbstractFloat && push!(r, norm(fx - y)); fx)
    res = profile_inversion(logged, prob.xa, prob.meas, prob.spectra, prob.setup)

    # relative change in the residual norm at each iteration, as used by the stopping test
    δ = abs.(diff(r)) ./ r[1:end-1]
    n_iter = length(δ)
    @test 2 <= n_iter < 30                # 30 is max_iter
    @test δ[end] <= 1e-6                  # it stopped because the fit converged ...
    @test all(>(1e-6), δ[2:end-1])        # ... and not before
    @test 0.5 < res.χ² < 1.5
end

@testset "cached cross-sections" begin
    # the profile operators reproduce the layer-by-layer loop (still used by the LHR model)
    prob = profile_problem()
    fields = collect(keys(prob.xa))
    x = assemble_state_vector!(prob.xa)
    xd = SF.assemble_state_vector!(x, fields, 2, prob.setup)
    p, T = prob.meas.pressure, prob.meas.temperature
    col = prob.setup["fit_column"]
    grid = prob.spectra[fields[1]].grid
    A = SF.layer_optical_depth_operators(prob.spectra, p, T, input_is_column=col)
    @test SF.calculate_transmission(xd, A) ≈ SF.calculate_transmission(xd, prob.spectra, p, T, grid, input_is_column=col)

    # the profile model reuses its cache, and rebuilds it when the measurement's p or T change
    f = generate_profile_model(prob.xa, prob.meas, prob.spectra, prob.setup)
    y = f(x)
    @test f(x) == y
    prob.meas.pressure = 0.9 .* p
    y_new = f(x)
    @test !(y_new ≈ y)
    @test y_new ≈ generate_profile_model(prob.xa, prob.meas, prob.spectra, prob.setup)(x)

    # same for the single-layer model
    x1 = gas_state(true_state())
    meas, spectra = synthetic_measurement(x1; σ=0.0)
    setup = column_setup()
    g = generate_forward_model(x1, meas, spectra, setup)
    y1 = g(x1)
    @test g(x1) == y1
    meas.temperature -= 10
    @test !(g(x1) ≈ y1)
    @test g(x1) ≈ generate_forward_model(x1, meas, spectra, setup)(x1)
end

@testset "multi-window (LHR) forward model" begin
    p, T = [850.0, 500.0], [290.0, 260.0]
    meas = make_measurement(zeros(length(MEAS_GRID)); p=p, T=T)
    windows = ((6078.0, 6079.5), (6080.0, 6082.0))
    setup = column_setup(poly_degree=[2, 2])
    spectra = setup_molecules(MOLECULES)

    # with no absorbers each window returns its own baseline
    x0 = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => [0.0, 0.0], "CH4" => [0.0, 0.0], "shape_parameters" => [1.0, 0.0, 2.0, 0.0])
    f = SF.generate_lhr_model(x0, meas, spectra, windows, setup)
    y0 = f(x0)
    n1 = length(SF.find_measurement_grid(windows[1], meas.grid))
    n2 = length(SF.find_measurement_grid(windows[2], meas.grid))
    @test length(y0) == n1 + n2
    @test y0[1:n1] ≈ ones(n1)
    @test y0[n1+1:end] ≈ fill(2.0, n2)        # second window uses its own coefficients

    # vector and dictionary inputs agree, and absorption lowers the signal
    vcd = make_vcd_profile(p, T)
    x = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "H2O" => 0.005 .* vcd, "CH4" => 1900e-9 .* vcd, "shape_parameters" => [1.0, 0.0, 2.0, 0.0])
    y = f(x)
    @test f(assemble_state_vector!(x)) ≈ y
    @test all(y .<= y0 .+ 1e-12) && any(y .< 0.99 .* y0)
    @test all(isfinite, ForwardDiff.jacobian(f, assemble_state_vector!(x)))
end

@testset "end-to-end: HDF5 file → averaged retrievals → JLD2" begin
    mktempdir() do dir
        data_dir = mkpath(joinpath(dir, "data"))
        out_dir = mkpath(joinpath(dir, "out"))
        truth = write_synthetic_dcs(joinpath(data_dir, "20160921.h5"); n_times=30, σ=2e-5)
        write(joinpath(data_dir, "notes.txt"), "not a data file")   # must be skipped

        setup = column_setup(averaging_window=Minute(15))
        vcd = vcd_surface()
        xa = StateVector("H2O" => 0.012 * vcd, "CH4" => 1700e-9 * vcd,
                         "shape_parameters" => [1.0, 0.0, 0.0])

        @test process_all_files(xa, MOLECULES, setup, [CH4_WINDOW];
                                data_path=data_dir, out_path=out_dir)
        outfile = joinpath(out_dir, "20160921_results.jld2")
        @test isfile(outfile)
        results = JLD2.load(outfile, "results")
        @test size(results) == (2, 1)          # two 15-minute averages, one window
        for r in results
            @test r isa InversionResults
            @test abs(r.x["CH4"] - 1900e-9 * vcd) < 4 * sqrt(r.S[2, 2])
            @test r.x["CH4"] / vcd ≈ 1900e-9 rtol = 0.05
            @test 0.5 < r.χ² < 2.0
        end

        # explicit file list variant
        out2 = mkpath(joinpath(dir, "out2"))
        @test process_all_files(xa, MOLECULES, setup, [CH4_WINDOW], ["20160921.h5"];
                                data_path=data_dir, out_path=out2)
        @test isfile(joinpath(out2, "20160921_results.jld2"))
    end
end

@testset "fit_spectra returns FailedInversion instead of throwing" begin
    mktempdir() do dir
        path = joinpath(dir, "bad.h5")
        write_synthetic_dcs(path; n_times=3)
        ds = read_DCS_data(path)
        setup = column_setup()
        # an H2O-only state makes the CH4 lookup inside the forward model fail
        xa = StateVector("H2O" => 1.0, "shape_parameters" => [1.0, 0.0, 0.0])
        spectra = setup_molecules(MOLECULES)
        f = x -> x
        res = fit_spectra(1, f, xa, ds, spectra, CH4_WINDOW, setup)
        @test res isa FailedInversion
        @test all(isnan, res.model)
    end
end
