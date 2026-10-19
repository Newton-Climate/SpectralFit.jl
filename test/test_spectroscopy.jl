# Unit tests for src/spectroscopy.jl against the vSmartMOM absorption module

@testset "get_molecule_info (HITRAN)" begin
    @test CH4_INFO isa MolecularMetaData
    @test CH4_INFO.molecule == "CH4"
    @test CH4_INFO.molecule_num == 6 && CH4_INFO.isotope_num == 1
    @test CH4_INFO.grid == MODEL_GRID
    @test CH4_INFO.model isa HitranModel

    spectra = setup_molecules(MOLECULES)
    @test collect(keys(spectra)) == ["H2O", "CH4"]
    @test spectra["CH4"] === CH4_INFO
end

@testset "construct_spectra" begin
    spec = construct_spectra(MOLECULES; ν_grid=MODEL_GRID, p=P_SURF, T=T_SURF)
    @test collect(keys(spec)) == ["H2O", "CH4"]
    for (name, mol) in spec
        @test mol isa Molecule
        @test length(mol.cross_sections) == length(MODEL_GRID)
        @test all(mol.cross_sections .>= 0)
        @test maximum(mol.cross_sections) > 0
    end

    # strongest CH4 line in the fixture window sits near 6079.1 cm⁻¹
    σ = spec["CH4"].cross_sections
    ν_peak = MODEL_GRID[argmax(σ)]
    @test 6077 < ν_peak < 6083
    @test σ ≈ absorption_cross_section(CH4_INFO.model, MODEL_GRID, P_SURF, T_SURF)

    # lower pressure narrows lines: the peak gets taller and the line core narrower
    spec_lowp = construct_spectra!(deepcopy(spec); p=300.0, T=T_SURF)
    @test spec_lowp isa AbstractDict
    σ_lowp = spec_lowp["CH4"].cross_sections
    @test maximum(σ_lowp) > maximum(σ)
    @test count(>(maximum(σ_lowp) / 2), σ_lowp) < count(>(maximum(σ) / 2), σ)
    @test σ_lowp ≈ absorption_cross_section(CH4_INFO.model, MODEL_GRID, 300.0, T_SURF)
end

@testset "calculate_cross_sections" begin
    mol = calculate_cross_sections(FIXTURE_PAR, 6, 1; ν_min=6077, ν_max=6083, δν=0.01,
                                   p=P_SURF, T=T_SURF)
    @test mol isa Molecule
    @test mol.p == P_SURF && mol.T == T_SURF
    @test mol.cross_sections ≈ absorption_cross_section(CH4_INFO.model, MODEL_GRID, P_SURF, T_SURF)

    calculate_cross_sections!(mol; p=300.0, T=T_SURF)
    @test mol.cross_sections ≈ absorption_cross_section(CH4_INFO.model, MODEL_GRID, 300.0, T_SURF)
end

@testset "construct_spectra_by_layer" begin
    p, T = [800.0, 600.0], [285.0, 265.0]
    layers = construct_spectra_by_layer(MOLECULES; p=p, T=T, ν_min=6077, ν_max=6083, δν=0.01)
    @test length(layers) == 2
    @test layers[1]["CH4"].p == 800.0 && layers[2]["CH4"].p == 600.0

    construct_spectra_by_layer!(layers; p=[700.0, 500.0], T=T)
    @test layers[2]["CH4"].cross_sections ≈
          absorption_cross_section(CH4_INFO.model, layers[2]["CH4"].grid, 500.0, 265.0)
end

@testset "get_molecule_info (interpolation table)" begin
    hitran = read_hitran(FIXTURE_PAR; mol=6, iso=1, ν_min=6077, ν_max=6083)
    p_grid = 700.0:100.0:900.0
    T_grid = 270.0:10.0:300.0
    itp_model = make_interpolation_model(hitran, Voigt(), MODEL_GRID, p_grid, T_grid)
    mktempdir() do dir
        path = joinpath(dir, "ch4_table.jld2")
        save_interpolation_model(itp_model, path)
        info = get_molecule_info("CH4", path, MODEL_GRID)
        @test info.molecule_num == 6 && info.isotope_num == 1
        @test info.model isa InterpolationModel

        # at a table node the interpolated cross-sections reproduce the line-by-line ones
        σ_table = absorption_cross_section(info.model, MODEL_GRID, 800.0, 280.0)
        σ_lbl = absorption_cross_section(CH4_INFO.model, MODEL_GRID, 800.0, 280.0)
        @test σ_table ≈ σ_lbl rtol = 1e-3
    end
end
