using Test
using Random
using LinearAlgebra
using MiniOps

# helper: check <A x, y> ≈ <x, A' y>
function check_adjoint(A, x, y; tol=1e-10)
    lhs = dot(A * x, y)
    rhs = dot(x, A' * y)
    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps())
    @test err < tol
    return err
end

@testset "MiniOps adjoint tests" begin
    Random.seed!(0)

    # 1. Convolution operator
    h = randn(5)
    A = conv1d_op(h)
    x = randn(10)
    y = randn(length(A * x))
    err_conv = check_adjoint(A, x, y)
    @info "conv adjoint err" err_conv

    # 2. Sampling operator (2D)
    nx, ny = 20, 30
    full_size = (nx, ny)
    mask = rand(nx * ny) .> 0.7
    idx  = findall(mask)
    R = sampling_op(idx, full_size)

    X = randn(nx, ny)
    d = R * X
    z = randn(length(d))
    err_samp = check_adjoint(R, X, z)
    @info "sampling adjoint err" err_samp

    # 3. Scaling by scalar
    S = scaling_op(2.5)
    x3 = randn(50)
    y3 = randn(50)
    err_scale = check_adjoint(S, x3, y3)
    @info "scaling adjoint err" err_scale

    # 4. Diagonal scaling
    w = randn(50)
    D = diag_op(w)
    x4 = randn(50)
    y4 = randn(50)
    err_diag = check_adjoint(D, x4, y4)
    @info "diag adjoint err" err_diag

    # 5. FFT operator (1D)
    F = fft_op()
    x5 = randn(64)
    y5 = randn(64)
    err_fft1d = check_adjoint(F, x5, y5)
    @info "fft 1d adjoint err" err_fft1d

    # 6. FFT operator (2D)
    X6 = randn(16, 16)
    Y6 = randn(16, 16)
    err_fft2d = check_adjoint(F, X6, Y6)
    @info "fft 2d adjoint err" err_fft2d

    # 7. Composition L = R * F * S * A (simple 1D composition)
    h7 = randn(5)
    A7 = conv1d_op(h7)
    S7 = scaling_op(0.3)
    F7 = fft_op()

    nx7 = 20
    x7  = randn(nx7)
    y7  = randn(length(F7 * (S7 * (A7 * x7))))

    # sampling in Fourier domain with full size (length of forward result)
    full_size7 = (length(F7 * (S7 * (A7 * x7))),)
    mask7 = rand(length(full_size7[1])) .> 0.5
    idx7  = findall(mask7)
    R7 = sampling_op(idx7, full_size7)

    L = R7 * F7 * S7 * A7

    xL = x7
    yL = randn(length(L * xL))

    err_L = check_adjoint(L, xL, yL)
    @info "composition adjoint err" err_L
end

