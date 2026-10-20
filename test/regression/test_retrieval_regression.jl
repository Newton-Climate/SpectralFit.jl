# Phase 6 regression: retrievals against the v0.1 reference (test/regression/README.md,
# groups ch4_inv and prof_inv). The problems are rebuilt from the recorded inputs; the forward
# model itself is checked against v0.1 in test_forward_model_regression.jl.
using JLD2

let ref = JLD2.load(joinpath(@__DIR__, "reference_v0_1.jld2"))
    fine = ref["ch4_fm/fine_grid_start"]:ref["ch4_fm/fine_grid_step"]:ref["ch4_fm/fine_grid_stop"]
    range_cm = (first(fine), last(fine))
    absorbers = (H2O = Absorber(:H2O, fixture_voigt(:H2O, range_cm)),
                 CH4 = Absorber(:CH4, fixture_voigt(:CH4, range_cm)))
    rel(a, b) = maximum(abs.(a .- b) ./ abs.(b))

    # ------------------------------------------------------------ column retrieval (ch4_inv)
    # v0.1 nonlinear_inversion is a least-squares fit, so the prior precision is zero and
    # x_first_guess is only the starting point.
    g = "ch4_inv"
    nu = ref["ch4_fm/meas_grid"]
    y = ref["$g/y_measured"]
    noise_variance = ref["$g/noise_sigma"]^2
    m = Measurement(y = y, coordinate = nu, noise_variance = noise_variance)
    first_guess = ref["$g/x_first_guess"]
    x0 = ComponentVector(H2O = first_guess[1], CH4 = first_guess[2],
                         baseline = (window_1 = first_guess[3:end],))
    @test ref["$g/state_names"][1:2] == ["H2O", "CH4"]
    @test length(first_guess) - 2 == ref["$g/setup_poly_degree"]
    atm = HomogeneousPath(pressure_hPa = ref["ch4_fm/pressure_hPa"],
                          temperature_K = ref["ch4_fm/temperature_K"],
                          path_length_cm = ref["ch4_fm/pathlength_cm"])
    model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(ref["$g/setup_poly_degree"]),
                                LinearSpace(), ColumnDensity(), m, x0; wavenumber_fine_cm = fine)
    Se_inv = obs_precision(LinearSpace(), m)
    @test isapprox(diag(Se_inv), ref["$g/Se_inv_diag"]; rtol = 1e-12)
    zero_prior = zeros(length(x0), length(x0))
    problem = RetrievalProblem(model, y, Se_inv, x0, zero_prior)

    @testset "ch4_inv retrieved state, chi2 and covariance" begin
        # v0.1 made n_iterations = 4 Gauss-Newton updates and returned the state after the
        # last one. A tight Rodgers tolerance makes the new solver take the same 4 updates
        # (from the same first guess with the same model, Jacobian and Se), so the returned
        # states are directly comparable.
        result = solve(problem, GaussNewton(rodgers_tolerance = 1e-10))
        @test result.status == Converged
        @test result.n_iterations == ref["$g/n_iterations"]
        @test rel(collect(result.x), ref["$g/x_retrieved"]) < 1e-6
        # zero prior: identity averaging kernel, no prior influence
        # (scaled by the posterior sigma, since A[i, j] has units of x_i / x_j)
        sig = sqrt.(diag(result.posterior_covariance))
        @test maximum(abs, (result.averaging_kernel - I) .* sig' ./ sig) < 1e-6
        @test abs(result.prior_influence) < 1e-6

        # v0.1 evaluated chi2, S and K at the last pre-update iterate x_4 (README, semantics 7),
        # one step before the returned state; the new result evaluates them at the returned
        # state. Compare like with like: evaluate the new diagnostics at the recorded iterate
        # ch4_inv/iterates[:, end].
        x_pre = ref["$g/iterates"][:, end]
        F, K = model_and_jacobian(problem, x_pre)
        @test rel(K, ref["$g/K"]) < 1e-6
        chi2_pre = chi2_per_dof(observation_cost(y, F, Se_inv), length(y), length(x0))
        @test abs(chi2_pre / ref["$g/chi2"] - 1) < 1e-6
        S_pre = posterior_covariance(K, Se_inv, zero_prior)
        @test rel(S_pre, ref["$g/S"]) < 1e-6

        # At convergence the pre-update and returned states differ by far less than the
        # noise (v0.1 model and f(x_retrieved) differ by 2.5e-13), so the diagnostics the new
        # code reports at the returned state also agree with the reference to 1e-6.
        @test abs(result.chi2_obs_per_dof / ref["$g/chi2"] - 1) < 1e-6
        @test rel(result.posterior_covariance, ref["$g/S"]) < 1e-6
        @test maximum(abs, result.y_model - ref["$g/model_at_x_retrieved"]) < 1e-10
    end

    @testset "ch4_inv with the default tolerance" begin
        # The default Rodgers tolerance (0.01) stops one update earlier; the state is still
        # within 1e-6 relative of v0.1 (the 4th update is about 1e-10 relative).
        result = solve(problem, GaussNewton())
        @test result.status == Converged
        @test rel(collect(result.x), ref["$g/x_retrieved"]) < 1e-6
        @test abs(result.chi2_obs_per_dof / ref["$g/chi2"] - 1) < 1e-6
    end

    # ------------------------------------------------------------ profile retrieval (prof_inv)
    p = "prof_inv"
    @test ref["$p/gamma"] == 0.0
    atm_layers = LayeredColumn(pressure_hPa = ref["$p/pressure_hPa"],
                               temperature_K = ref["$p/temperature_K"],
                               solar_zenith_deg = ref["$p/sza_deg"])
    @test slant_factor(atm_layers, 1) == ref["$p/air_mass_factor"]
    y_p = ref["$p/y_measured"]
    m_p = Measurement(y = y_p, coordinate = ref["$p/meas_grid"],
                      noise_variance = ref["$p/noise_sigma"]^2)
    xp = ref["$p/x_prior"]
    x_prior = ComponentVector(H2O = xp[1:3], CH4 = xp[4:6], baseline = (window_1 = xp[7:9],))
    @test ref["$p/state_names"][[1, 4, 7]] == ["H2O[1]", "CH4[1]", "shape_parameters[1]"]
    model_p = build_forward_model(atm_layers, absorbers, DCS(), LegendreBaseline(ref["$p/setup_poly_degree"]),
                                  LinearSpace(), ColumnDensity(), m_p, x_prior;
                                  wavenumber_fine_cm = fine)
    Se_inv_p = obs_precision(LinearSpace(), m_p)
    Sa_inv_p = Diagonal(ref["$p/Sa_inv_diag"])
    problem_p = RetrievalProblem(model_p, y_p, Se_inv_p, x_prior, Sa_inv_p)

    @testset "prof_inv forward model" begin
        # v0.1 generate_profile_model at the recorded truth and at the returned state
        x_true = ComponentVector(ref["$p/x_true"], getaxes(x_prior))
        @test maximum(abs, model_p(x_true) - ref["$p/y_clean"]) < 1e-10
        x_ret = ComponentVector(ref["$p/x_retrieved"], getaxes(x_prior))
        @test maximum(abs, model_p(x_ret) - ref["$p/model"]) < 1e-10
    end

    # "Within the convergence tolerance": a solver with Rodgers tolerance t stops when the last
    # step dx has dx' N dx / n_state < t (N the normal matrix), so its answer can differ from
    # the exact maximum a posteriori state by about that much in the same metric. The test
    # measures the distance to the v0.1 state in that metric and requires it below t.
    function within_convergence_tolerance(result, x_ref, tolerance)
        dx = collect(result.x) - x_ref
        N = normal_matrix(result.K, Se_inv_p, Sa_inv_p)
        return dot(dx, N * dx) / length(dx) < tolerance
    end

    solvers = (("Gauss-Newton", GaussNewton()),
               ("LM fixed damping gamma = 0 (v0.1 profile_inversion)",
                LevenbergMarquardt(damping = FixedDamping(ref["$p/gamma"]))),
               ("LM fixed damping gamma = 1", LevenbergMarquardt(damping = FixedDamping(1.0))),
               ("LM adaptive damping", LevenbergMarquardt(damping = AdaptiveDamping())))
    for (label, solver) in solvers
        @testset "prof_inv state, $label" begin
            result = solve(problem_p, solver)
            @test result.status == Converged
            @test within_convergence_tolerance(result, ref["$p/x_retrieved"], solver.rodgers_tolerance)
            # issue 1: posterior covariance from the corrected formula (with the prior term)
            # at the solution; the reference has no posterior covariance to compare with
            ax = getaxes(x_prior)
            K = ForwardDiff.jacobian(v -> model_p(ComponentVector(v, ax)), collect(result.x))
            S = inv(Symmetric(Matrix(K' * Se_inv_p * K) + Matrix(Sa_inv_p)))
            s = 1 ./ sqrt.(ref["$p/Sa_inv_diag"])          # prior sigma, for unit-free scaling
            @test isapprox(result.posterior_covariance ./ (s * s'), S ./ (s * s'); rtol = 1e-8)
        end
    end

    @testset "prof_inv matches v0.1 iterations exactly" begin
        # Gauss-Newton with the default tolerance and LM with gamma = 0 make the same 3 MAP
        # updates as v0.1 profile_inversion (n_iterations = 3, tolerance 1e-6 on the relative
        # change of the residual norm), so the states agree to rounding.
        for solver in (GaussNewton(), LevenbergMarquardt(damping = FixedDamping(0.0)))
            result = solve(problem_p, solver)
            @test result.n_iterations == ref["$p/n_iterations"]
            @test rel(collect(result.x), ref["$p/x_retrieved"]) < 1e-6
            # v0.1 profile chi2 is evaluated at the returned state, without the prior term
            @test abs(result.chi2_obs_per_dof / ref["$p/chi2"] - 1) < 1e-6
        end
    end
end
