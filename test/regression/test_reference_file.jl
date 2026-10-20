using JLD2
@testset "v0.1 reference file loads" begin
    ref = JLD2.load(joinpath(@__DIR__, "reference_v0_1.jld2"))
    @test haskey(ref, "ch4_fm_intensity/spectrum")
    @test haskey(ref, "ch4_inv/x_retrieved")
    @test haskey(ref, "prof_inv/x_retrieved")
end
