struct Op{F,FT}
    f::F
    ft::FT
    m::Int
    n::Int
    name::Symbol
end

Op(f, ft; m, n, name=:anonymous) = 
    Op{typeof(f),typeof(ft)}(f, ft, m, n, name)

Base.size(A::Op) = (A.m, A.n)
(A::Op)(x) = A.f(x)
Base.:*(A::Op, x) = A.f(x)


# Build a "shaped" operator B from A using a test input x0
# m = number of elements of A*x0, n = number of elements of x0
function with_shape(A::Op, x0)
    y0 = A * x0
    m  = length(vec(y0))
    n  = length(vec(x0))
    return Op(A.f, A.ft; m = m, n = n, name = A.name)
end