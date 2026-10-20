@testset "parameter slots" begin
    x = ComponentVector(a = 2.0, b = [1.0, 2.0], instrument = (stretch = 1.5,))
    @test resolve(Fixed(3.0), x) == 3.0
    @test resolve(FromState{:a}(), x) == x.a
    @test resolve(FromState{(:instrument, :stretch)}(), x) == 1.5
    @test @inferred(resolve(Fixed(3.0), x)) == 3.0
    @test @inferred(resolve(FromState{:a}(), x)) == 2.0
    @test @inferred(resolve(FromState{(:instrument, :stretch)}(), x)) == 1.5
    @test has_state_path(x, :a)
    @test has_state_path(x, (:instrument, :stretch))
    @test !has_state_path(x, :pressure_hPa)
    @test !has_state_path(x, (:a, :b))
    @test bind_parameter(Fixed(1.0), :a, x) === FromState{:a}()
    @test bind_parameter(Fixed(1.0), :c, x) === Fixed(1.0)
    @test_throws ArgumentError bind_parameter(FromState{:c}(), :c, x)
    @test as_parameter(2.0) === Fixed(2.0)
    @test as_parameter(FromState{:a}()) === FromState{:a}()
end

@testset "GasConditions promotes" begin
    c = GasConditions(850, 290.0, 0)
    @test c isa GasConditions{Float64}
    d = ForwardDiff.Dual(850.0, 1.0)
    @test GasConditions(d, 290.0, 0.0) isa GasConditions{typeof(d)}
end
