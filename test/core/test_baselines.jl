# Tests for src/core/baselines.jl: Legendre matrix, baselines in every mode and space,
# measurement spaces, observation precision and masks.

baseline_dual_state(x) = ComponentVector(ForwardDiff.Dual{Nothing}.(collect(x), 1.0), getaxes(x))

baseline_test_measurement(n = 50) =
    Measurement(y = 1.0 .+ 0.5 .* sin.(range(0, 3; length = n)),
                coordinate = collect(range(6076.0, 6078.0; length = n)),
                noise_variance = fill(1e-4, n))

@testset "legendre_matrix pinned to v0.1 compute_legendre_poly" begin
    # values from v0.1 compute_legendre_poly(collect(range(-1, 1, length = 7)), 5), transposed
    pinned = [
        1.0 -1.0 1.0 -1.0 1.0
        1.0 -0.6666666666666666 0.16666666666666663 0.2592592592592593 -0.42746913580246915
        1.0 -0.3333333333333333 -0.33333333333333337 0.40740740740740744 0.012345679012345678
        1.0 0.0 -0.5 -0.0 0.375
        1.0 0.3333333333333333 -0.33333333333333337 -0.40740740740740744 0.012345679012345678
        1.0 0.6666666666666666 0.16666666666666663 -0.2592592592592593 -0.42746913580246915
        1.0 1.0 1.0 1.0 1.0
    ]
    @test legendre_matrix(7, 5) == pinned
    @test legendre_matrix(7, 1) == ones(7, 1)
    @test legendre_matrix(7, 2) == pinned[:, 1:2]
    @test_throws ArgumentError legendre_matrix(7, 0)
    @test_throws ArgumentError legendre_matrix(1, 3)
end

@testset "legendre_matrix orthogonality" begin
    n = 20001
    P = legendre_matrix(n, 6)
    w = fill(2 / (n - 1), n); w[1] /= 2; w[end] /= 2     # trapezoid weights on [-1, 1]
    G = P' * Diagonal(w) * P
    expected = Diagonal([2 / (2l + 1) for l in 0:5])
    @test maximum(abs.(G .- expected)) < 1e-6
end

@testset "measurement spaces" begin
    m = baseline_test_measurement()
    y_before = copy(m.y)
    @test to_measurement_space(LinearSpace(), m.y) === m.y
    @test to_measurement_space(LogSpace(), m.y) == log.(m.y)
    @test observation_vector(LinearSpace(), m) == m.y
    @test observation_vector(LinearSpace(), m) !== m.y
    @test observation_vector(LogSpace(), m) == log.(m.y)
    @test m.y == y_before
end

@testset "every baseline and space combination" begin
    m = baseline_test_measurement()
    y = 0.9 .+ 0.05 .* cos.(range(0, 5; length = 50))
    for n in (1, 4), space in (LinearSpace(), LogSpace()), mode in (Multiplicative(), Additive())
        b = LegendreBaseline(n; mode = mode, name = :w1)
        @test n_parameters(b) == n
        @test baseline_name(b) == :w1
        pb = prepare(b, m)
        @test pb isa PreparedLegendreBaseline{typeof(mode),:w1}
        @test n_parameters(pb) == n
        @test pb.matrix == legendre_matrix(50, n)
        c = [1.2, -0.3, 0.2, 0.1][1:n]
        x = ComponentVector(CH4 = 1.0, baseline = (w0 = [9.0], w1 = c))
        ys = to_measurement_space(space, y)
        term = legendre_matrix(50, n) * c
        expected = (mode isa Multiplicative && space isa LinearSpace) ? ys .* term : ys .+ term
        @test apply_baseline(pb, space, ys, x) == expected
        @test (@inferred apply_baseline(pb, space, ys, x)) isa Vector{Float64}
        xd = baseline_dual_state(x)
        @test (@inferred apply_baseline(pb, space, ys, xd)) isa Vector{eltype(xd)}
        @test (@inferred apply_baseline(pb, space, ForwardDiff.Dual{Nothing}.(ys, 1.0), x)) isa
              Vector{ForwardDiff.Dual{Nothing,Float64,1}}
        @test_throws ArgumentError apply_baseline(b, space, ys, x)
    end
    # one constant coefficient: multiplicative linear scales, additive offsets
    x1 = ComponentVector(baseline = (window_1 = [2.0],))
    @test apply_baseline(prepare(LegendreBaseline(1), m), LinearSpace(), y, x1) == 2 .* y
    @test apply_baseline(prepare(LegendreBaseline(1; mode = Additive()), m), LinearSpace(), y, x1) == y .+ 2
    @test apply_baseline(prepare(LegendreBaseline(1), m), LogSpace(), log.(y), x1) == log.(y) .+ 2
    # no baseline
    for space in (LinearSpace(), LogSpace())
        @test prepare(NoBaseline(), m) === NoBaseline()
        @test n_parameters(NoBaseline()) == 0
        @test apply_baseline(NoBaseline(), space, y, x1) === y
        @test (@inferred apply_baseline(NoBaseline(), space, y, baseline_dual_state(x1))) === y
        @test initial_coefficients(NoBaseline(), space, m) == Float64[]
    end
    @test_throws ArgumentError LegendreBaseline(0)
    @test_throws ArgumentError LegendreBaseline(3; mode = LinearSpace())
    @test_throws DimensionMismatch prepare(prepare(LegendreBaseline(2), m), baseline_test_measurement(10))
end

@testset "initial coefficients" begin
    m = baseline_test_measurement()
    ymax = maximum(m.y)
    @test initial_coefficients(LegendreBaseline(3), LinearSpace(), m) == [ymax, 0.0, 0.0]
    @test initial_coefficients(LegendreBaseline(3), LogSpace(), m) == [log(ymax), 0.0, 0.0]
    @test initial_coefficients(LegendreBaseline(1), LinearSpace(), m) == [ymax]
    @test initial_coefficients(prepare(LegendreBaseline(2), m), LinearSpace(), m) == [ymax, 0.0]
    @test initial_coefficients(LegendreBaseline(2; mode = Additive()), LinearSpace(), m) == [0.0, 0.0]
end

@testset "obs_precision in linear and log space" begin
    n = 50
    m = Measurement(y = collect(range(0.2, 3.0; length = n)),
                    coordinate = collect(range(6076.0, 6078.0; length = n)),
                    noise_variance = collect(range(1e-4, 4e-4; length = n)))
    S_lin = obs_precision(LinearSpace(), m)
    S_log = obs_precision(LogSpace(), m)
    @test S_lin isa Diagonal && S_log isa Diagonal
    @test S_lin.diag == 1 ./ m.noise_variance
    # first-order propagation from the stored linear y (log y would be negative for y < 1)
    @test S_log.diag == m.y .^ 2 ./ m.noise_variance
    @test all(>(0), S_log.diag)
    bad = Measurement(y = [1.0, -0.1], coordinate = [1.0, 2.0], noise_variance = 1.0)
    @test_throws ArgumentError obs_precision(LogSpace(), bad)
end

@testset "LogSpace precision matches Monte Carlo variance" begin
    rng = MersenneTwister(20261019)
    y, sigma, n = 2.0, 0.01, 200_000
    samples = log.(y .+ sigma .* randn(rng, n))
    v = var(samples)
    standard_error = v * sqrt(2 / (n - 1))
    m = Measurement(y = [y], coordinate = [6076.0], noise_variance = sigma^2)
    predicted = 1 / obs_precision(LogSpace(), m)[1, 1]
    @test predicted == sigma^2 / y^2
    @test abs(v - predicted) < 3 * standard_error
end

@testset "masked windows are de-weighted" begin
    nu = collect(6076.0:0.1:6078.0)
    m = Measurement(y = fill(2.0, length(nu)), coordinate = nu, noise_variance = 1e-4)
    windows = [(6076.25, 6076.55), (6077.65, 6078.5)]
    idx = SpectralFits.CoreLayer.find_mask(nu, windows)
    @test idx == [4, 5, 6, 18, 19, 20, 21]
    @test SpectralFits.CoreLayer.find_mask(nu, [6076.25 6076.55; 6077.65 6078.5]) == idx
    @test isempty(SpectralFits.CoreLayer.find_mask(nu, [(7000.0, 7001.0)]))
    # open intervals: a point on the window edge is not masked
    @test SpectralFits.CoreLayer.find_mask(nu, [(6076.0, 6076.2)]) == [2]
    for space in (LinearSpace(), LogSpace())
        full = obs_precision(space, m).diag
        masked = obs_precision(space, m; masked_windows_cm = windows).diag
        @test masked[idx] == full[idx] .* 1e-5
        keep = setdiff(eachindex(nu), idx)
        @test masked[keep] == full[keep]
        @test obs_precision(space, m; masked_windows_cm = windows, mask_factor = 0.5).diag[idx] ==
              full[idx] .* 0.5
    end
    @test_throws ArgumentError obs_precision(LinearSpace(), m; mask_factor = 0.0)
end
