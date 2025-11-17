module MiniOps

using LinearAlgebra      # Standard LA library
using FFTW               # FFTs

export Op,
       conv1d_op,
       conv1d_cols_op,
       sampling_op,
       scaling_op,
       diag_op,
       fft_op,
       with_shape, 
       adjoint_test,
       linearity_test,
       opnorm_power,
       is_selfadjoint

# 1) Define Op first
include("core.jl")

# 2) Then methods that depend on Op (adjoint, composition, etc.)
include("algebra.jl")

# 3) Then concrete operators
include("conv.jl")
include("sampling.jl")
include("scaling.jl")
include("fftops.jl")

# 4) Diagnostics using Op and the operators
include("diagnostics.jl")

end # module


