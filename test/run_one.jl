# Run selected test files: julia --project=test test/run_one.jl test/core/test_a.jl [more files]
include("helpers.jl")
@testset "selected" begin
    for file in ARGS
        @testset "$file" begin
            include(abspath(file))
        end
    end
end
