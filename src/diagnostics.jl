using LinearAlgebra

# Check <A x, y> ≈ <x, A' y> for given test arrays x,y
function adjoint_test(A::Op, x, y; tol=1e-10)
    Ax  = A * x
    Aty = A' * y

    lhs = dot(vec(Ax), vec(y))
    rhs = dot(vec(x),  vec(Aty))

    err = abs(lhs - rhs) / max(abs(lhs), abs(rhs), eps())
    return (err < tol, err)
end




# Check linearity: A(a x + b z) ≈ a A x + b A z
# x and z must have the same shape and eltype, matching the domain of A.
function linearity_test(A::Op, x, z; ntests=3, tol=1e-10)
    maxerr = 0.0
    for _ in 1:ntests
        a = randn()          # real scalars are fine even for complex x,z
        b = randn()

        lhs = A * (a .* x .+ b .* z)
        rhs = a .* (A * x) .+ b .* (A * z)

        err = norm(vec(lhs - rhs)) / max(norm(vec(lhs)), norm(vec(rhs)), eps())
        maxerr = max(maxerr, err)
        err < tol || return (false, maxerr)
    end
    return (true, maxerr)
end

# returns ‖A‖₂ (spectral norm) of OP
function opnorm_power(A::Op, x0; niter=20)
    x = copy(x0)
    x ./= norm(vec(x))

    λ = zero(real(eltype(x)))  # scalar accumulator
    for _ in 1:niter
        # apply A'A to x
        y = A' * (A * x)

        # Rayleigh-like iterate: ||A'A x||
        λ = norm(vec(y))

        # normalize next iterate
        if λ == 0
            return 0.0
        end
        x .= y ./ λ
    end

    # here λ ~ λ_max(A'A) = ||A||^2, specral norm is sqrt(λ)
    return sqrt(λ)
end



# Test self-adjointness on a given test array x
function is_selfadjoint(A::Op, x; tol=1e-10)
    lhs = A * x
    rhs = A' * x

    err = norm(vec(lhs - rhs)) / max(norm(vec(lhs)), norm(vec(rhs)), eps(real(float(norm(lhs)))))
    return (err < tol, err)
end
