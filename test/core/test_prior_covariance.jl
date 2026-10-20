# Correlated prior covariance and the scaled inverse.

@testset "correlated_prior_covariance structure" begin
    z = collect(0.0:0.5:4.5)                                  # km, 10 layers
    sigma = ComponentVector(H2O = fill(2e21, 10), CH4 = fill(5e17, 10), baseline = (window_1 = ones(3),))
    Sa = correlated_prior_covariance(sigma, z, (CH4 = 1.0,); nugget = 0.0)
    ich4 = state_indices(sigma, :CH4)
    @test Sa[ich4[1], ich4[2]] ≈ 5e17^2 * exp(-0.5^2 / 2) rtol = 1e-14
    @test Sa[ich4[1], ich4[1]] ≈ 5e17^2 rtol = 1e-14
    ih2o = state_indices(sigma, :H2O)
    @test Sa[ih2o[1], ih2o[2]] == 0                           # H2O not correlated
    @test Sa[ih2o[1], ich4[1]] == 0                           # gases not cross-correlated
    @test issymmetric(Sa)
    @test diag(correlated_prior_covariance(sigma, z, (CH4 = 1.0,); nugget = 1e-3))[ich4] ≈ (1 + 1e-3) * 5e17^2 .* ones(10)
    @test_throws r"layers" correlated_prior_covariance(sigma, z[1:5], (CH4 = 1.0,))
    @test_throws r"no entry" correlated_prior_covariance(sigma, z, (CO2 = 1.0,))
end

@testset "correlated_prior_precision inverts a nearly singular Gaussian kernel" begin
    # 35 layers, 50 hPa correlation on layers about 20 hPa apart: the kernel alone has a
    # condition number near 1e16, the nugget makes it invertible
    p = 1000 .* exp.(-collect(range(0, 10_000, length = 35)) ./ 8500)
    sigma = ComponentVector(CO2 = fill(4e19, 35), baseline = (window_1 = ones(5),))
    P = correlated_prior_precision(sigma, p, (CO2 = 50.0,))
    Sa = correlated_prior_covariance(sigma, p, (CO2 = 50.0,))
    d = 1 ./ collect(sigma)
    @test issymmetric(P)
    @test opnorm((d .* Sa .* d') * (P ./ (d .* d')) - I) < 1e-6        # inverse in scaled units
    @test isposdef(Symmetric(P ./ (d .* d')))
end

@testset "scaled_inverse keeps the small entries of a badly scaled matrix" begin
    rng = MersenneTwister(3)
    B = randn(rng, 6, 6)
    M = B' * B + 6I                                          # well conditioned
    s = 10.0 .^ [-11, -8, 0, 3, 9, 11]
    N = Symmetric(s .* M .* s')                              # unscaled condition near 1e44
    S = scaled_inverse(N)
    @test S ≈ (1 ./ s) .* inv(M) .* (1 ./ s)' rtol = 1e-10
    @test_throws PosDefException scaled_inverse(Symmetric(-Matrix(1.0I, 2, 2)))
end

@testset "column_averaging_kernel and state_indices" begin
    x = ComponentVector(H2O = [1.0, 2.0], CH4 = [3.0, 4.0, 5.0], baseline = (window_1 = [1.0],))
    @test state_indices(x, :CH4) == [3, 4, 5]
    @test state_indices(x, :baseline) == [6]
    A = Matrix(1.0I, 6, 6)
    @test column_averaging_kernel(A, [3, 4, 5], [1.0, 2.0, 3.0]) ≈ ones(3)
    @test_throws DimensionMismatch column_averaging_kernel(A, [3, 4], [1.0, 2.0, 3.0])
end
