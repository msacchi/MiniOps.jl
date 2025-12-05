# 1D forward convolution: y = conv(x, h)
function forward_conv(x::AbstractVector, h::AbstractVector)
    nx = length(x)
    nh = length(h)
    y = similar(x, nx + nh - 1)
    fill!(y, zero(eltype(x)))

    for ix = 1:nx
        for ih = 1:nh
            iy = ix + ih - 1
            y[iy] += x[ix] * h[ih]
        end
    end
    return y
end

# Adjoint of 1D convolution with respect to x
# (transpose of Toeplitz matrix built from h)
function adjoint_conv(y::AbstractVector, h::AbstractVector)
    ny = length(y)
    nh = length(h)
    nx = ny - nh + 1
    x = similar(y, nx)
    fill!(x, zero(eltype(y)))

    for ix = 1:nx
        for ih = 1:nh
            iy = ix + ih - 1
            x[ix] += y[iy] * h[ih]
        end
    end
    return x
end


"""
    conv1d_op(h)

Return a MiniOps `Op` for 1D convolution/crosscorrelation with vector h.

Domain (model):  x  :: AbstractVector  (nx x 1)

Range (data):    y  :: AbstractVector  (ny × 1)

h is lenght nh and x is lenght nx, then y is lenght nx+ny-1

If `C = conv1d_op(h)`, then 

Forward: `y = C * x`        # Convolution of h with h         

Adjoint: `xa = C' * y`      # Crosscorrelation of h with y
"""
function conv1d_op(h::AbstractVector)
    hvec = collect(h)
    f  = x -> forward_conv(x, hvec)
    ft = y -> adjoint_conv(y, hvec)
    # m,n are not known statically; we set them to -1 as "unknown"
    return Op(f, ft; m = -1, n = -1, name = :conv1d)
end

# --------------------------------------------------
# Multi-trace convolution along columns of a matrix

# X: (nt, ntr) -> Y: (nt + nh - 1, ntr)
function forward_conv_cols(X::AbstractMatrix, h::AbstractVector)
    nt, ntr = size(X)
    nh      = length(h)
    ny      = nt + nh - 1

    Y = zeros(promote_type(eltype(X), eltype(h)), ny, ntr)

    for itr = 1:ntr
        for ix = 1:nt
            for ih = 1:nh
                iy = ix + ih - 1
                Y[iy, itr] += X[ix, itr] * h[ih]
            end
        end
    end
    return Y
end

# Y: (nt + nh - 1, ntr) -> X: (nt, ntr)
function adjoint_conv_cols(Y::AbstractMatrix, h::AbstractVector)
    ny, ntr = size(Y)
    nh      = length(h)
    nt      = ny - nh + 1

    X = zeros(promote_type(eltype(Y), eltype(h)), nt, ntr)

    for itr = 1:ntr
        for ix = 1:nt
            for ih = 1:nh
                iy = ix + ih - 1
                X[ix, itr] += Y[iy, itr] * h[ih]
            end
        end
    end
    return X
end

"""
    conv1d_cols_op(h)

Return a MiniOps `Op` for 1D multichannel convolution/crosscorrelation with 
vector h. It repeats 1D conv/crosscorrelation over columns.

Domain (model):  X  :: AbstractMatrix  (nx x ntraces)

Range (data):    Y  :: AbstractMatrix (ny × traces)

h is lenght nh and x is (nx x ntraces), then y is (nx+ny-1 x ntraces)

If `C = conv1d_cols_op(h)`, then

Forward:  `Y = C * X`          # Convolution of h with columns of X        

Adjoint:  `Xa = C' * Y`        # Crosscorrelation of h with columns of Y
"""
function conv1d_cols_op(h::AbstractVector)
    hvec = collect(h)

    f  = X -> forward_conv_cols(X, hvec)
    ft = Y -> adjoint_conv_cols(Y, hvec)

    return Op(f, ft; m = -1, n = -1, name = :conv1d_cols)
end
