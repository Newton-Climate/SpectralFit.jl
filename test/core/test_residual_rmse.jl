# Residual RMSE in the retrieval result.

@testset "residual_rmse leaves out de-weighted observations" begin
    y = [1.0, 2.0, 3.0, 4.0]
    y_model = [1.1, 1.9, 3.0, 10.0]
    P = Diagonal([1e6, 1e6, 1e6, 1e6 * 1e-5])                 # last observation masked
    @test residual_rmse(y, y_model, P) ≈ sqrt((0.01 + 0.01 + 0.0) / 3)
    @test residual_rmse(y, y_model, Diagonal(fill(1e6, 4))) ≈ sqrt((0.01 + 0.01 + 36.0) / 4)
    @test full_weight_observations(P) == [true, true, true, false]
end

@testset "RetrievalResult carries the RMSE" begin
    K = [1.0 0.0; 0.0 1.0; 1.0 1.0]
    y = [1.0, 2.0, 3.5]
    model = x -> K * collect(x)
    x0 = ComponentVector(a = 0.0, b = 0.0)
    problem = RetrievalProblem(model, y, Diagonal(fill(100.0, 3)), x0, Diagonal([1e-6, 1e-6]))
    r = solve(problem, GaussNewton())
    @test r.rmse ≈ sqrt(sum(abs2, y .- r.y_model) / 3)
    @test occursin("residual RMSE", report(r))
end
