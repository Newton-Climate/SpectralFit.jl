# Phase 2 tests for load_cross_section_model (HITRAN .par files and JLD2 lookup tables).
import vSmartMOM.Absorption as Absorption
import JLD2

@testset "load_cross_section_model(HitranVoigt)" begin
    load = SF.IOLayer.load_cross_section_model
    grid = 6225.0:0.01:6231.0
    c = GasConditions(850.0, 290.0, 0.0)

    co2 = load(HitranVoigt, FIXTURE_PAR; molecule = 2, isotope = 1, range_cm = (6224.0, 6232.0))
    @test co2 isa HitranVoigt
    @test spectral_range_cm(co2) == (6224.0, 6232.0)
    @test all(==(2), co2.model.hitran.mol)
    @test all(==(1), co2.model.hitran.iso)
    @test all(nu -> 6224.0 <= nu <= 6232.0, co2.model.hitran.νᵢ)
    @test co2.model.wing_cutoff == 40

    # same lines and options as reading with vSmartMOM directly
    table = Absorption.read_hitran(FIXTURE_PAR; mol = 2, iso = 1, ν_min = 6224.0, ν_max = 6232.0)
    direct = Absorption.make_hitran_model(table, Absorption.Voigt(); wing_cutoff = 40)
    @test cross_section(co2, grid, c) ==
          Absorption.absorption_cross_section(direct, grid, 850.0, 290.0)

    # each molecule of the subset loads, in the window that holds it
    h2o = load(HitranVoigt, FIXTURE_PAR; molecule = 1, isotope = 1, range_cm = (6076.0, 6084.0))
    @test all(==(1), h2o.model.hitran.mol)
    @test maximum(cross_section(h2o, 6077.0:0.01:6083.0, c)) > 0

    # options are passed through to vSmartMOM
    narrow = load(HitranVoigt, FIXTURE_PAR; molecule = 2, isotope = 1, range_cm = (6224.0, 6232.0),
                  wing_cutoff_cm = 5, broadening = Absorption.Lorentz(), self_vmr = 0.01)
    @test narrow.model.wing_cutoff == 5
    @test narrow.model.broadening isa Absorption.Lorentz
    @test narrow.model.vmr == 0.01
    # a margin reads lines outside the range; the valid range is unchanged
    wide = load(HitranVoigt, FIXTURE_PAR; molecule = 6, isotope = 1, range_cm = (6078.0, 6082.0),
                line_margin_cm = 2.0)
    @test spectral_range_cm(wide) == (6078.0, 6082.0)
    @test minimum(wide.model.hitran.νᵢ) < 6078.0 && maximum(wide.model.hitran.νᵢ) > 6082.0

    @test_throws ArgumentError load(HitranVoigt, "no_such_file.par"; molecule = 6,
                                    range_cm = (6076.0, 6084.0))
    @test_throws ArgumentError load(HitranVoigt, FIXTURE_PAR; molecule = 6,
                                    range_cm = (6150.0, 6160.0))   # no lines there
    @test_throws ArgumentError load(HitranVoigt, FIXTURE_PAR; molecule = 6,
                                    range_cm = (6084.0, 6076.0))
end

@testset "load_cross_section_model(LookupTable)" begin
    load = SF.IOLayer.load_cross_section_model
    table = Absorption.read_hitran(FIXTURE_PAR; mol = 6, iso = 1, ν_min = 6076.0, ν_max = 6084.0)
    itp_model = Absorption.make_interpolation_model(table, Absorption.Voigt(), 6078.0:0.02:6082.0,
                                                    800.0:50.0:900.0, 280.0:10.0:300.0)
    mktempdir() do dir
        path = joinpath(dir, "ch4_lut.jld2")
        JLD2.jldsave(path; itp_model)
        lut = load(LookupTable, path)
        @test lut isa LookupTable
        @test spectral_range_cm(lut) == (6078.0, 6082.0)
        @test lut.pressure_range_hPa == (800.0, 900.0)
        @test lut.temperature_range_K == (280.0, 300.0)
        grid = 6079.0:0.01:6081.0
        c = GasConditions(850.0, 290.0, 0.0)
        @test cross_section(lut, grid, c) == cross_section(LookupTable(itp_model), grid, c)

        other = joinpath(dir, "other.jld2")
        JLD2.jldsave(other; something_else = [1.0, 2.0])
        @test_throws ArgumentError load(LookupTable, other)
        @test_throws ArgumentError load(LookupTable, other; key = "something_else")
        @test_throws ArgumentError load(LookupTable, joinpath(dir, "missing.jld2"))
    end
end
