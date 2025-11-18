
function soft_threshold(u, tau)
    # tau >= 0 (scalar)
    amp    = abs.(u)
    denom  = amp .+ eps(Float64)   # avoid division by zero
    factor = max.(0.0, 1 .- tau ./ denom)
    return factor .* u
end


# ISTA: solves
#   min_u  0.5 * ||A*u - y||_2^2 + mu * ||u||_1
#
# Inputs:
#   A         - MiniOps.Op operator
#   y         - data (same shape as A*u)
#   u0        - initial model (same shape as solution u)
#   mu        - L1 penalty weight
#   step_size - gradient step size
#   niter     - number of iterations
#
# Output:
#   u         - ISTA solution
function ista(A, y, u0, mu, step_size; niter=100, verbose=false)
    u = copy(u0)

    for k in 1:niter
        # Gradient of 0.5 * ||A*u - y||^2
        r = A * u .- y
        g = A' * r

        # Gradient step on L2 term
        u .= u .- step_size .* g

        # Prox step (soft-threshold) for L1
        u .= soft_threshold(u, mu * step_size)

        if verbose && (k % 20 == 0)
            @show k, maximum(abs.(r))
        end
    end

    return u
end



"""
cgls(A, b, mu, x0; tol=1e-6, max_iter=1000)

Solve the quadratic regularized least-squares problem via the Conjugate Gradient Least Squares (CGLS) method.

# Arguments:
- A: The matrix in the least-squares problem.
- b: The right-hand side vector.
- mu: The regularization parameter.
- x0: The initial guess for the solution.
- tol: The tolerance for convergence (default is 1e-6).
- max_iter: The maximum number of iterations (default is 1000).

# Returns:
- x: The computed solution.
"""
function cgls(A, b, mu, x0; tol=1e-6, max_iter=1000)
    r = b - A * x0           # Initial residual
    s = A' * r               # Initial search direction
    p = s                    # Set p to initial search direction
    old_inner_product = dot(s, s)  # Inner product of s with itself
    x = x0                   # Initialize solution

    for k in 1:max_iter
        q = A * p            # Compute A*p
        alpha = old_inner_product / (dot(q, q) + mu * dot(p, p))  # Step size
        x += alpha * p       # Update solution
        r -= alpha * q       # Update residual
        s = A' * r - mu * x  # Compute new gradient
        new_inner_product = dot(s, s)  # Inner product of new s with itself
        
        # Check for convergence
        if sqrt(new_inner_product) < tol
            break
        end
        
        beta = new_inner_product / old_inner_product  # Compute beta
        p = s + beta * p            # Update search direction
        old_inner_product = new_inner_product  # Update old inner product
    end

    return x  # Return the computed solution
end


