# Phase 6: profile retrieval on a LayeredColumn (four layers, CH4 and H2O per layer, Legendre
# baseline) with Levenberg-Marquardt, fixed and adaptive damping. The synthetic measurement is
# made by its own forward model on a fine grid ten times finer than the retrieval grid
# (inverse-crime guard).

let
    range_cm = (6076.0, 6084.0)
    absorbers = (H2O = Absorber(:H2O, fixture_voigt(:H2O, range_cm)),
                 CH4 = Absorber(:CH4, fixture_voigt(:CH4, range_cm)))
    pressure_hPa = [900.0, 700.0, 500.0, 300.0]
    temperature_K = [288.0, 275.0, 255.0, 235.0]
    atm = LayeredColumn(pressure_hPa = pressure_hPa, temperature_K = temperature_K,
                        solar_zenith_deg = 40.0)
    n_layer = SF.CoreLayer.column_profile(atm, nothing)   # vertical dry-air columns, molecules/cm2
    nu = collect(6078.0:0.0067:6082.0)
    retrieval_grid = 6077.0:0.005:6083.0
    truth_grid = 6077.0:0.0005:6083.0
    noise_sigma = 1e-4

    x_true = ComponentVector(H2O = [0.008, 0.004, 0.0015, 0.0004] .* n_layer,
                             CH4 = [1950e-9, 1900e-9, 1850e-9, 1780e-9] .* n_layer,
                             baseline = (window_1 = [1.0, 0.01, -0.005],))
    m_template = Measurement(y = ones(length(nu)), coordinate = nu, noise_variance = noise_sigma^2)
    truth_model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                                      ColumnDensity(), m_template, x_true;
                                      wavenumber_fine_cm = truth_grid)
    y = truth_model(x_true) .+ noise_sigma .* randn(MersenneTwister(7), length(nu))
    m = Measurement(y = y, coordinate = nu, noise_variance = noise_sigma^2)

    x_prior = ComponentVector(H2O = 0.004 .* n_layer, CH4 = fill(1800e-9, 4) .* n_layer,
                              baseline = (window_1 = [maximum(y), 0.0, 0.0],))
    sigma = ComponentVector(H2O = 0.005 .* n_layer, CH4 = fill(300e-9, 4) .* n_layer,
                            baseline = (window_1 = [1.0, 1.0, 1.0],))
    model = build_forward_model(atm, absorbers, DCS(), LegendreBaseline(3), LinearSpace(),
                                ColumnDensity(), m, x_prior; wavenumber_fine_cm = retrieval_grid)
    Se_inv = obs_precision(LinearSpace(), m)
    Sa_inv = prior_precision_from_sigma(sigma, x_prior)
    problem = RetrievalProblem(model, observation_vector(LinearSpace(), m), Se_inv, x_prior, Sa_inv)

    total_true = sum(x_true.CH4)
    prior_error = abs(sum(x_prior.CH4) / total_true - 1)
    ch4_index = findall(startswith("CH4"), labels(x_prior))

    @test maximum(abs, model(x_true) - truth_model(x_true)) > 1e-9    # no inverse crime

    for (label, solver) in (("fixed damping", LevenbergMarquardt(damping = FixedDamping(1.0))),
                            ("adaptive damping", LevenbergMarquardt(damping = AdaptiveDamping())))
        @testset "profile retrieval, $label" begin
            result = solve(problem, solver)
            @test result.status == Converged
            total = sum(result.x.CH4)
            error = abs(total / total_true - 1)
            @test error < 0.01
            @test error < prior_error
            @test prior_error > 0.03                     # the prior was really off
            # total column within 4 posterior sigma
            h = zeros(length(result.x)); h[ch4_index] .= 1
            @test abs(total - total_true) < 4 * sqrt(h' * result.posterior_covariance * h)
            @test 0.7 < result.chi2_obs_per_dof < 1.3

            # issue 1: posterior covariance includes the prior term, at the solution
            ax = getaxes(x_prior)
            K = ForwardDiff.jacobian(v -> model(ComponentVector(v, ax)), collect(result.x))
            S_formula = inv(Symmetric(Matrix(K' * Se_inv * K) + Matrix(Sa_inv)))
            @test isapprox(result.K, K; rtol = 1e-10)
            # compare in prior-sigma units, so columns (1e22 molecules/cm2) and baseline
            # coefficients (order 1) weigh alike
            s = collect(sigma)
            @test isapprox(result.posterior_covariance ./ (s * s'), S_formula ./ (s * s');
                           rtol = 1e-8)
            # averaging kernel A = I - S Sa_inv, and the prior matters (DFS below n_state)
            n = length(result.x)
            A_formula = I - S_formula * Matrix(Sa_inv)
            @test maximum(abs, (result.averaging_kernel - A_formula) .* s' ./ s) < 1e-6
            @test result.dfs_total < n - 0.5
            @test isapprox(result.dfs_total, tr(result.averaging_kernel); rtol = 1e-12)
        end
    end
end
