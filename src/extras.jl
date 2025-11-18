function seismic_wavelet(;f0 = 20.0, dt = 0.002)
    N = floor(Int,(3.0/f0)/dt)
    w = (exp.(-6.0*collect(0:1:N)/N)).*sin.(2*pi*f0*collect(0:1:N)*dt)
    M = floor(Int,0.2*N)
    w = edge_cosine_taper(w,M);
return w/norm(w)
end


function edge_cosine_taper(h::AbstractVector, m::Int)
    n = length(h)
    @assert 2m ≤ n "Taper length m too large for wavelet length"
    w = ones(Float64, n)

    t = range(0, π/2, length=m)
    left  = sin.(t).^2
    right = reverse(left)

    w[1:m] .= left
    w[end-m+1:end] .= right

    return h .* w
end
