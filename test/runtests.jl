using Test

include("helpers.jl")

@testset "SpectralFits" begin
    @testset "unit: utils" begin
        include("test_utils.jl")
    end
    @testset "unit: read_data" begin
        include("test_read_data.jl")
    end
    @testset "unit: spectroscopy" begin
        include("test_spectroscopy.jl")
    end
    @testset "integration: retrievals" begin
        include("test_integration.jl")
    end
end
