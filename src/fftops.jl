# FFT operator on any array (1D, 2D, ND)
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
