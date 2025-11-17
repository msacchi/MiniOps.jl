# MiniOps.jl

A lightweight, matrix-free linear operator framework for inverse problems, signal processing, seismic imaging, and PDE/FWI kernels.

`MiniOps` allows you to write expressions such as:

```julia
y  = A * x
x̂  = A' * y
L  = R * F * A
z  = L * x
```

where `A`, `F`, `R`, … are linear **operators**, not matrices.

The goals:

- Minimal but powerful abstraction  
- Matrix-free forward/adjoint operators  
- Shape-agnostic (1D, 2D, ND arrays)  
- Real/complex type-agnostic  
- Clean algebra: composition, adjoint, scalar multiplication  
- Essential operators:
  - 1D convolution
  - columnwise convolution
  - sampling
  - scaling & diagonal ops
  - FFT / inverse FFT
- Diagnostics: adjoint test, linearity test, norm estimate

---

# 1. Installation & Usage

Inside the package folder:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Then:

```julia
using MiniOps
```

For development:

```julia
using Pkg
Pkg.develop(path="/path/to/MiniOps")
```

---

# 2. Core Concept: `Op`

An `Op` represents a matrix-free linear operator:

```julia
struct Op{F,FT}
    f::F      # forward  x ↦ A*x
    ft::FT    # adjoint  y ↦ A'*y
    m::Int    # number of elements in codomain (optional)
    n::Int    # number of elements in domain   (optional)
    name::Symbol
end
```

### Key ideas:

- `f`/`ft` are closures (forward/adjoint).
- `m` and `n` default to `-1` (shape unknown).
- Operators compose:

```julia
L = R * F * A
```

- Adjoint works automatically:

```julia
At = A'
```

- Operators work on arrays of **any shape**.

### Primary overloads:

| Expression           | Meaning                         |
|----------------------|---------------------------------|
| `A * x`              | forward                         |
| `A(x)`               | same as above                   |
| `A' * y`             | adjoint                         |
| `C = A * B`          | composition                     |
| `size(A)`            | returns `(m,n)`                 |
| `α * A`              | scalar multiplication           |

---

# 3. Helper: `with_shape`

Assign meaningful `(m,n)` based on a sample input:

```julia
A_shaped = with_shape(A, x0)
```

Sets:

- `n = length(vec(x0))`
- `m = length(vec(A*x0))`

Example:

```julia
F = fft_op()
X0 = randn(ComplexF64,128,128)
F = with_shape(F, X0)
@show size(F)
```

---

# 4. Built-in Operators

## 4.1 `conv1d_op(h)`
1D convolution operator.

```julia
A = conv1d_op(h)
y = A * x
```

---

## 4.2 `conv1d_cols_op(h)`
Columnwise convolution (applied to each trace).

```julia
A = conv1d_cols_op(h)
Y = A * X
Z = A' * Y
```

---

## 4.3 `sampling_op(idx, full_size)`
Generalized sampling (masking): forward extracts samples, adjoint scatters back.

---

## 4.4 `scaling_op(α)`
Scalar multiplication operator.

---

## 4.5 `diag_op(w)`
Diagonal operator: `y = w .* x`

---

## 4.6 `fft_op()`
Matrix-free FFT/bFFT operator.

```julia
F = fft_op()
Y = F * X
X2 = F' * Y
```

---

# 5. Diagnostics

## 5.1 `adjoint_test(A, x, y)`
Validates inner-product identity:

\[
\langle Ax, y \rangle \approx \langle x, A'y \rangle.
\]

---

## 5.2 `linearity_test(A, x, z)`
Checks linearity.

---

## 5.3 `opnorm_power(A, x0)`
Power-method estimate of ‖A‖₂.

---

## 5.4 `is_selfadjoint(A, x)`
Tests whether `A ≈ A'`.

---

# 6. Example: Linear Inverse Problem

```julia
using MiniOps

nx, ny = 128, 128
full_size = (nx, ny)

h = [1.0, 2.0, -1.0]
A = conv1d_cols_op(h)
F = fft_op()

mask = rand(nx*ny) .> 0.6
idx  = findall(mask)
R = sampling_op(idx, full_size)

L = R * F * A

x_true = zeros(ComplexF64,nx,ny)
x_true[2,4] = 1 + 2im
x_true[4,3] = -1 - 1im

y = L * x_true
```

---

# 7. Extending MiniOps

```julia
function my_op()
    f  = x -> forward_code(x)
    ft = y -> adjoint_code(y)
    Op(f, ft; m=-1, n=-1, name=:my_op)
end
```

Run:

```julia
adjoint_test(A, randn(size), randn(size))
```

---

# 8. Running Tests

```julia
using Pkg
Pkg.test("MiniOps")
```

---

# 9. Design Notes

- Matrix-free  
- Shape-agnostic  
- Type-agnostic  
- Explicit algebra  
- Ideal for FWI, tomography, deconvolution, compressed sensing  

