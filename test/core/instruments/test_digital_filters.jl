# Butterworth design and filtering. Reference numbers for order 5, cutoff 100 Hz, fs 20 kHz
# were computed independently with scipy 1.x:
#     b, a = scipy.signal.butter(5, 100.0, fs=20000.0)
#     scipy.signal.sosfilt(scipy.signal.butter(5, 100.0, fs=20000.0, output='sos'), ones(400))
# The order-2 case is checked against the textbook closed form of the bilinear-transformed
# prototype with K = tan(pi fc / fs).

@testset "Butterworth design" begin
    @test_throws ArgumentError Butterworth(order = 0, cutoff_Hz = 1.0)
    @test_throws ArgumentError design_filter(Butterworth(order = 2, cutoff_Hz = 600.0), 1000.0)

    for order in (1, 2, 3, 5, 8), (fc, fs) in ((100.0, 20_000.0), (50.0, 1000.0), (3e3, 1e4))
        sos = design_filter(Butterworth(order = order, cutoff_Hz = fc), fs)
        @test abs2(frequency_response(sos, fc)) ≈ 0.5 rtol = 1e-12      # -3 dB at the cutoff
        @test frequency_response(sos, 0.0) ≈ 1.0 rtol = 1e-12
        # digital magnitude equals the analogue prototype at the prewarped frequency
        for f in (0.1fc, 0.7fc, 1.3fc, 0.45fs)
            analog = 1 / sqrt(1 + (tan(pi * f / fs) / tan(pi * fc / fs))^(2order))
            @test abs(frequency_response(sos, f)) ≈ analog rtol = 1e-10
        end
    end

    # order 2: closed-form biquad
    fc, fs = 50.0, 1000.0
    K = tan(pi * fc / fs)
    D = 1 + sqrt(2) * K + K^2
    sos = design_filter(Butterworth(order = 2, cutoff_Hz = fc), fs)
    @test sos.b[:, 1] ≈ [K^2, 2K^2, K^2] ./ D rtol = 1e-13
    @test sos.a[:, 1] ≈ [2(K^2 - 1), 1 - sqrt(2) * K + K^2] ./ D rtol = 1e-13

    # order 5 pinned against scipy.signal.butter
    sos = design_filter(Butterworth(order = 5, cutoff_Hz = 100.0), 20_000.0)
    b, a = transfer_function_coefficients(sos)
    b_ref = [9.0928661148194707e-10, 4.5464330574097356e-09, 9.0928661148194712e-09,
             9.0928661148194712e-09, 4.5464330574097356e-09, 9.0928661148194707e-10]
    a_ref = [1.0, -4.898337145711599, 9.598497090805596, -9.405307989195732,
             4.608476358536906, -0.9033282853379996]
    @test b ≈ b_ref rtol = 1e-9
    @test a ≈ a_ref rtol = 1e-12
    # section denominators (scipy pairs the same poles)
    @test sort(sos.a[1, :]) ≈ sort([-9.6906741719379330e-01, -1.9494734182360649e+00,
                                    -1.9797963102817411e+00]) rtol = 1e-12
    @test sort(sos.a[2, :]) ≈ [0.0, 9.5043584058390662e-01, 9.8077370253308749e-01] rtol = 1e-12
end

@testset "Butterworth filtering" begin
    sos = design_filter(Butterworth(order = 5, cutoff_Hz = 100.0), 20_000.0)
    step_response = filter_signal(sos, ones(400))
    @test step_response[[1, 2, 3, 100, 400]] ≈
          [9.0928661148194707e-10, 9.9097120540119343e-09, 5.4361911546165854e-08,
           3.6951784511830749e-01, 1.0148569675760324e+00] rtol = 1e-9
    @test filter_signal(sos, ones(20_000))[end] ≈ 1.0 rtol = 1e-10

    # a sinusoid well inside the pass band keeps its amplitude after the transient
    fs = 20_000.0
    t = (0:39_999) ./ fs
    y = filter_signal(sos, sin.(2pi * 10.0 .* t))
    H = frequency_response(sos, 10.0)
    @test maximum(abs, y[20_001:end] .- abs(H) .* sin.(2pi * 10.0 .* t[20_001:end] .+ angle(H))) < 1e-6

    # dual numbers pass through and the derivative is the filtered derivative (linearity)
    g(s) = filter_signal(sos, s .* sin.(2pi * 30.0 .* t[1:2000]))
    J = ForwardDiff.derivative(g, 1.7)
    @test J ≈ filter_signal(sos, sin.(2pi * 30.0 .* t[1:2000])) rtol = 1e-12
    @test eltype(filter_signal(sos, ForwardDiff.Dual.(ones(3), 1.0))) <: ForwardDiff.Dual
end
