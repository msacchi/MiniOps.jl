# Sampling operator: picks entries at idx from an array of size full_size.
# Works with any array x of size full_size (1D, 2D, ...).

function sampling_op(idx::AbstractVector{<:Integer}, full_size::NTuple{N,Int}) where {N}
    idx_vec = collect(idx)
    nfull   = prod(full_size)
    m       = length(idx_vec)

    # Forward: sample entries
    f = function (x)
        x_vec = vec(x)
        return x_vec[idx_vec]
    end

    # Adjoint: scatter back (zero-fill)
    ft = function (y)
        Y = zeros(eltype(y), nfull)
        Y[idx_vec] .= y
        return reshape(Y, full_size)
    end

    return Op(f, ft; m = m, n = nfull, name = :sampling)
end


