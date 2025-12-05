"""
    fft_op(; unitary = true)
    fft_op(x0; unitary = true)

Construct a MiniOps `Op` that applies the ND discrete Fourier transform
(DFT) and its adjoint.

There are two variants:

  • `fft_op(; unitary = true)`

    Returns an operator that uses `fft` / `bfft` directly. This is a
    convenient, non-planned FFT operator that can be applied to any
    vector `x` (length determined at apply time):

    ```julia
    F = fft_op()          # or fft_op(unitary = false)
    y = F * x             # forward FFT
    xrec = F' * y         # adjoint (inverse-side transform)
    ```

    Since the length is not fixed at construction, the operator sizes
    `m, n` are set to `-1` as “unknown” placeholders.

  • `fft_op(x0; unitary = true)`

    Returns an operator that uses FFTW plans created from the prototype
    vector `x0`. This is more efficient when applying the FFT repeatedly
    to vectors of the same length and element type:

    ```julia
    F = fft_op(x0)        # plans fft and bfft for length(x0)
    y = F * x             # x must have length(x0)
    xrec = F' * y
    ```

    In this case, the operator is square with fixed size
    `m = n = length(x0)`.

### Scaling convention

If `unitary = true` (default):

  - Forward: `y = fft(x) / sqrt(M)`
  - Adjoint: `x = bfft(y) / sqrt(M)`

where `M = length(x)` (or `length(x0)` for the planned version).
"""
function fft_op(; unitary = true)
    if unitary
        f  = x -> begin
            M = length(x)
            fft(x) ./ sqrt(M)
        end
        ft = y -> begin
            M = length(y)
            bfft(y) ./ sqrt(M)
        end
    else
        f  = x -> fft(x)
        ft = y -> bfft(y)
    end

    return Op(f, ft; m = -1, n = -1, name = :fft_op)
end



function fft_op(x0; unitary = true)
    M     = length(x0)
    planF = plan_fft(x0)
    planB = plan_bfft(x0)

    if unitary
        f  = x -> (planF * x) ./ sqrt(M)
        ft = y -> (planB * y) ./ sqrt(M)
    else
        f  = x -> planF * x
        ft = y -> planB * y
    end

    return Op(f, ft; m = M, n = M, name = :fft_planned)
end



function pad_nd(x::AbstractArray,
                pad_before::NTuple{N,Int},
                pad_after::NTuple{N,Int}) where {N}

    @assert ndims(x) == N
    @assert length(pad_before) == N == length(pad_after)

    out_sizes = ntuple(i -> pad_before[i] + size(x, i) + pad_after[i], N)
    y = zeros(eltype(x), out_sizes)

    ranges = ntuple(i -> (pad_before[i] + 1):(pad_before[i] + size(x, i)), N)
    @inbounds y[ranges...] .= x

    return y
end

function unpad_nd(y::AbstractArray,
                  pad_before::NTuple{N,Int},
                  pad_after::NTuple{N,Int}) where {N}

    @assert ndims(y) == N
    @assert length(pad_before) == N == length(pad_after)

    in_sizes = ntuple(i -> size(y, i) - pad_before[i] - pad_after[i], N)
    @assert all(s -> s > 0, in_sizes)

    ranges = ntuple(i -> (pad_before[i] + 1):(pad_before[i] + in_sizes[i]), N)

    x = similar(y, eltype(y), in_sizes)
    @inbounds x .= @view y[ranges...]

    return x
end

"""
    pad_op(sz_in, pad_before, pad_after)

Create an N-D padding operator for arrays of size `sz_in`.

- `sz_in`      :: NTuple{N,Int}
- `pad_before` :: NTuple{N,Int}
- `pad_after`  :: NTuple{N,Int}
"""
function pad_op(sz_in::NTuple{N,Int},
                pad_before::NTuple{N,Int},
                pad_after::NTuple{N,Int}) where {N}

    @assert length(pad_before) == N == length(pad_after)

    # Output size of padded array
    sz_out = ntuple(i -> pad_before[i] + sz_in[i] + pad_after[i], N)

    # Flattened sizes for Op bookkeeping
    n = prod(sz_in)   # input dof
    m = prod(sz_out)  # output dof

    f = function (x)
        @assert size(x) == sz_in
        pad_nd(x, pad_before, pad_after)
    end

    ft = function (y)
        @assert size(y) == sz_out
        unpad_nd(y, pad_before, pad_after)
    end

    return Op(f, ft; m = m, n = n, name = :pad_op)
end

# Convenience: pass a prototype array instead of sz_in
pad_op(x0::AbstractArray,
       pad_before::NTuple{N,Int},
       pad_after::NTuple{N,Int}) where {N} =
    pad_op(size(x0), pad_before, pad_after)


