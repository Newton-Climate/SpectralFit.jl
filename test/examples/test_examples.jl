# The example scripts in examples/ run and give sensible answers (keeps README.md honest).
# dcs_merlin_batch.jl needs real data and is not run.

function run_example(name)
    path = joinpath(pkgdir(SpectralFits), "examples", name)
    mod = Module(Symbol(splitext(name)[1]))
    Core.eval(mod, :(include(x) = Base.include($mod, x)))
    redirect_stdout(devnull) do
        Base.include(mod, path)
    end
    return mod
end

@testset "examples/quickstart_dcs.jl" begin
    ex = run_example("quickstart_dcs.jl")
    @test ex.result.status == Converged
    @test abs(ex.xch4_ppb - 1950) < 3 * ex.sigma_ppb
end

@testset "examples/profile_retrieval.jl" begin
    ex = run_example("profile_retrieval.jl")
    @test ex.result.status == Converged
    @test abs(ex.total - sum(ex.x_true.CH4)) < 4 * ex.sigma
end

@testset "examples/wms_retrieval.jl" begin
    ex = run_example("wms_retrieval.jl")
    @test ex.result.status == Converged
    @test abs(ex.result.x.CH4 - ex.x_true.CH4) < 4 * sqrt(ex.result.posterior_covariance[1, 1])
end
