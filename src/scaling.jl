"""
    scaling_op(alpha::Number) -> Op

Create a linear operator that multiplies its input by a scalar `alpha`.

The forward and adjoint actions are

    y = alpha .* x

for any array `x`. This operator works for vectors or multidimensional
arrays and preserves the shape of the input.

Arguments
---------
alpha : Scalar multiplier.

Returns
-------
Op
    A MiniOps operator representing scalar multiplication.

Notes
-----
- For real `alpha`, the operator is self-adjoint.
- For complex `alpha`, the mathematical adjoint would normally apply
  the complex conjugate. This implementation applies the same scalar
  to both forward and adjoint.
- Operator dimensions are set to `-1` because they depend on the size
  of the input passed at apply time.
"""
function scaling_op(alpha::Number)
    f  = x -> alpha .* x
    ft = y -> alpha .* y
    return Op(f, ft; m = -1, n = -1, name = :scaling)
end

"""
    diag_op(w) -> Op

Create a diagonal linear operator with weights `w`.

The operator acts elementwise as

    y = w .* x

where `w` and `x` must have compatible sizes.

Arguments
---------
w : Vector or array of weights defining the diagonal entries.

Returns
-------
Op
    A MiniOps operator implementing elementwise scaling.

Notes
-----
- For real-valued `w`, the forward and adjoint operators are the same.
- For complex-valued `w`, the true adjoint should apply the complex
  conjugate of `w`; this implementation applies the same weights in
  both directions.
- Operator dimensions are set to `-1` because they depend on the size
  of the input used at runtime.
"""
function diag_op(w)
    wcopy = copy(w)
    f  = x -> wcopy .* x
    ft = y -> wcopy .* y
    return Op(f, ft; m = -1, n = -1, name = :diag)
end


