# Unit tests for src/utils.jl and the covariance helpers in src/inversion.jl

@testset "column densities" begin
    # Loschmidt number: molecules per cm³ at 1013.25 hPa and 273.15 K
    @test SF.calc_vcd(1013.25, 273.15, 1.0) ≈ 2.6868e19 rtol = 1e-3

    # number density scales linearly with path length and dry fraction
    @test SF.calc_vcd(850.0, 290.0, 2.0) ≈ 2 * SF.calc_vcd(850.0, 290.0, 1.0)
    @test SF.calc_vcd(850.0, 290.0, 1.0, 0.02) ≈ 0.98 * SF.calc_vcd(850.0, 290.0, 1.0)

    # hydrostatic dry column for a 100 hPa layer: Δp / (m_dry g), in molec/cm²
    m_dry = 28.9647e-3 / SF.Nₐ
    @test SF.vcd_pressure(100.0, 280.0, 0.0) ≈ 100.0 * 100 / (m_dry * SF.g₀) / 1e4
end

@testset "half_pressure_levels and make_vcd_profile" begin
    # uniformly spaced levels give uniform layer thickness, including the boundaries
    @test SF.half_pressure_levels([1000.0, 900.0, 800.0]) ≈ [100.0, 100.0, 100.0]

    p = [900.0, 700.0, 500.0]
    T = [290.0, 270.0, 250.0]
    vcd = make_vcd_profile(p, T)
    @test length(vcd) == 3
    @test all(vcd .> 0)
    @test vcd ≈ [SF.vcd_pressure(200.0, t, 0.0) for t in T]
end

@testset "StateVector" begin
    x = StateVector("H2O" => 1.0, "CH4" => 2.0, "shape_parameters" => [3.0, 4.0])
    @test x isa OrderedDict{String,Union{Float64,Vector{Float64}}}
    @test collect(keys(x)) == ["H2O", "CH4", "shape_parameters"]
end

@testset "assemble_state_vector! single level" begin
    x = StateVector("H2O" => 1.0, "CH4" => 2.0, "pressure" => 850.0,
                    "shape_parameters" => [3.0, 4.0, 5.0])
    v = assemble_state_vector!(x)
    @test v == [1.0, 2.0, 850.0, 3.0, 4.0, 5.0]

    back = assemble_state_vector!(v, collect(keys(x)), RetrievalSetup("poly_degree" => 3))
    @test back == x

    # no shape parameters
    y = StateVector("H2O" => 1.0, "CH4" => 2.0)
    @test assemble_state_vector!(assemble_state_vector!(y), collect(keys(y)), RetrievalSetup()) == y
end

@testset "assemble_state_vector! multiple levels" begin
    x = OrderedDict{String,Union{Float64,Vector{Float64}}}(
        "CH4" => [1.0, 2.0], "H2O" => [3.0, 4.0], "shape_parameters" => [5.0, 6.0, 7.0])
    v = assemble_state_vector!(x)
    @test v == collect(1.0:7.0)
    back = assemble_state_vector!(v, collect(keys(x)), 2, RetrievalSetup("poly_degree" => 3))
    @test back == x

    # per-window polynomial degrees are summed
    back2 = assemble_state_vector!(v, collect(keys(x)), 2, RetrievalSetup("poly_degree" => [1, 2]))
    @test back2["shape_parameters"] == [5.0, 6.0, 7.0]

    # one level falls back to the single-level method
    z = StateVector("CH4" => 1.0, "shape_parameters" => [2.0, 3.0])
    @test assemble_state_vector!([1.0, 2.0, 3.0], collect(keys(z)), 1,
                                 RetrievalSetup("poly_degree" => 2)) == z
end

@testset "Legendre baseline" begin
    x = collect(range(-1, 1, length=11))
    P = SF.compute_legendre_poly(x, 4)
    @test size(P) == (4, 11)
    @test P[1, :] ≈ ones(11)
    @test P[2, :] ≈ x
    @test P[3, :] ≈ @. (3x^2 - 1) / 2
    @test P[4, :] ≈ @. (5x^3 - 3x) / 2

    coeffs = [1.0, 0.2, -0.1]
    term = calc_polynomial_term(3, coeffs, 11)
    @test vec(term) ≈ @. 1.0 + 0.2x - 0.1 * (3x^2 - 1) / 2
end

@testset "apply_instrument" begin
    fine = 6000.0:0.01:6010.0
    signal = sin.(fine ./ 2)
    coarse = collect(6001.003:0.0371:6009.0)
    @test apply_instrument(fine, signal, coarse) ≈ sin.(coarse ./ 2) atol = 1e-8
end

@testset "index and mask helpers" begin
    grid = collect(1.0:10.0)
    @test SF.find_indexes(2.0, 5.0, grid) == [3, 4]          # bounds are exclusive
    @test SF.find_measurement_grid((2.0, 5.0), grid) == [3.0, 4.0]
    @test SF.find_mask(grid, [1.5 3.5; 7.5 9.5]) == [2, 3, 8, 9]
end

@testset "make_obs_error" begin
    m = make_measurement([1.0, 2.0, 4.0, 1.0]; grid=[1.0, 2.0, 3.0, 4.0], σ²=0.01)
    @test SF.make_obs_error(m) ≈ Diagonal(fill(100.0, 4))
    @test SF.make_obs_error(m; σ²=0.04) ≈ Diagonal(fill(25.0, 4))
    # linear mode: intensity holds log(I), so the weight is I²/σ² with I = exp(intensity)
    m_log = make_measurement(log.([1.0, 2.0, 4.0, 1.0]); grid=[1.0, 2.0, 3.0, 4.0], σ²=0.01)
    @test SF.make_obs_error(m_log; linear=true) ≈ Diagonal([1.0, 4.0, 16.0, 1.0] ./ 0.01)

    masked = SF.make_obs_error(m; masked_windows=[1.5 3.5])
    @test diag(masked) ≈ [100.0, 1 / (1e5 * 0.01), 1 / (1e5 * 0.01), 100.0]

    # windows outside the grid leave the covariance untouched
    @test SF.make_obs_error(m; masked_windows=[10.0 11.0]) ≈ Diagonal(fill(100.0, 4))
end

@testset "make_prior_error" begin
    @test SF.make_prior_error([1.0, 2.0]) ≈ Diagonal([1.0, 0.25])
    σ = StateVector("CH4" => 0.5, "shape_parameters" => [1.0, 4.0])
    @test SF.make_prior_error(σ) ≈ Diagonal([4.0, 1.0, 1 / 16])
end

@testset "shape-parameter priors" begin
    m = make_measurement([1.0, 3.0, 2.0, 5.0, 4.0]; grid=[1.0, 2.0, 3.0, 4.0, 5.0])
    setup = RetrievalSetup("poly_degree" => [2, 3])
    sp = make_shape_params(m, [(0.5, 2.5), (2.5, 5.5)], setup)
    @test sp == [3.0, 0.0, 5.0, 0.0, 0.0]

    ds = SF.FrequencyCombDataset(filename="", intensity=[1.0 7.0; 2.0 3.0], grid=[1.0, 2.0],
                                 temperature=[1.0, 1.0], pressure=[1.0, 1.0],
                                 time=[DateTime(2020), DateTime(2020)], pathlength=1.0,
                                 num_averaged_measurements=[1, 1], averaging_window=Second(1),
                                 machine_time=[0.0, 1.0], σ²=[1.0, 1.0], vcd=[1.0, 1.0])
    @test prior_shape_params(ds, RetrievalSetup("poly_degree" => 3)) == [7.0, 0.0, 0.0]
end

@testset "check_param and failed inversions" begin
    setup = RetrievalSetup("linear" => true)
    @test SF.check_param(setup, "linear") == true
    @test SF.check_param(setup, "missing_key") == false

    x = StateVector("CH4" => 1.0, "shape_parameters" => [1.0, 2.0])
    xf = SF.failed_vector(x)
    @test isnan(xf["CH4"]) && all(isnan, xf["shape_parameters"])
    @test x["CH4"] == 1.0 # input untouched

    m = make_measurement(ones(4); grid=collect(1.0:4.0))
    fi = SF.failed_inversion(x, m)
    @test fi isa FailedInversion
    @test isnan(fi.χ²) && all(isnan, fi.model)
end

@testset "calc_DCS_noise" begin
    rng = MersenneTwister(1)
    k = 1e-3                                 # noise σ = k * sqrt(signal)
    signal = collect(range(0.5, 2.0, length=40))
    intensity = signal .+ k .* sqrt.(signal) .* randn(rng, 40, length(NOISE_GRID))
    σ² = SF.calc_DCS_noise(NOISE_GRID, intensity)
    @test length(σ²) == 40
    @test σ² ≈ k^2 .* signal rtol = 0.15
end

@testset "gain matrix and χ²" begin
    rng = MersenneTwister(3)
    K = randn(rng, 20, 3)
    model = randn(rng, 20)
    meas = model .+ 0.1 .* randn(rng, 20)
    Se⁻¹ = Matrix(1.0I, 20, 20)
    res = InversionResults(timestamp=DateTime(2020), machine_time=0.0,
                           x=StateVector("a" => 1.0, "b" => 2.0, "c" => 3.0),
                           measurement=meas, model=model, χ²=0.0, S=zeros(3, 3),
                           grid=collect(1.0:20.0), K=K, Sₑ⁻¹=Se⁻¹, Sₐ⁻¹=zeros(3, 3))
    @test calc_gain_matrix(res) ≈ pinv(K)
    @test SF.calc_χ²(res) ≈ sum(abs2, model - meas) / 17
end
