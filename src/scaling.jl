# Scaling by a scalar alpha: y = alpha * x
function scaling_op(alpha::Number)
    f  = x -> alpha .* x
    ft = y -> alpha .* y
    return Op(f, ft; m = -1, n = -1, name = :scaling)
end

# Diagonal operator with weights w: y = w .* x
function diag_op(w)
    wcopy = copy(w)
    f  = x -> wcopy .* x
    ft = y -> wcopy .* y   # adjoint is the same for real w
    return Op(f, ft; m = -1, n = -1, name = :diag)
end


