# Runs every test/**/test_*.jl file, each in its own testset, in sorted path order.
# Run one file with: julia --project=test test/run_one.jl test/core/test_air_columns.jl

include("helpers.jl")

function find_test_files(root)
    files = String[]
    for (dir, _, names) in walkdir(root), name in names
        startswith(name, "test_") && endswith(name, ".jl") && push!(files, joinpath(dir, name))
    end
    return sort(files)
end

@testset "SpectralFits" begin
    for file in find_test_files(@__DIR__)
        @testset "$(relpath(file, @__DIR__))" begin
            include(file)
        end
    end
end
