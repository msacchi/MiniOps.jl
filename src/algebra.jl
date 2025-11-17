
# Adjoint
Base.adjoint(A::Op) = Op(A.ft, A.f; m = A.n, n = A.m,
                         name = Symbol(A.name, "'"))

# Composition: L = A * B
function Base.:*(A::Op, B::Op)
    # NOTE: no size check for now, because many ops are shape-agnostic
    f  = x -> A * (B * x)
    ft = y -> B' * (A' * y)
    return Op(f, ft; m = A.m, n = B.n,
              name = Symbol(A.name, "*", B.name))
end

# Scalar times operator
function Base.:*(c::Number, A::Op)
    f  = x -> c .* (A * x)
    ft = y -> c .* (A' * y)
    return Op(f, ft; m = A.m, n = A.n,
              name = Symbol(c, "*", A.name))
end
