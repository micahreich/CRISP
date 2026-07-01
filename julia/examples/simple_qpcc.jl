using Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CRISP

Q = [
    2.0 0.0 0.0 0.0
    0.0 2.0 0.0 0.0
    0.0 0.0 2.0 0.0
    0.0 0.0 0.0 2.0
]
q = zeros(4)

data = (
    Q = Q,
    q = q,
    c0 = 0.0,
    J_eq = [1.0 0.0 0.0 0.0],
    b_eq = [-1.0],
    J_ineq = [0.0 1.0 0.0 0.0],
    b_ineq = [-1.0],
    L = [0.0 0.0 1.0 0.0],
    l = [1.0],
    R = [0.0 0.0 0.0 1.0],
    r = [-1.0],
)

# No solver options are passed, so CRISP uses its C++ default parameters.
result = CRISP.solve_qpcc_with_crisp(data)

@show result.converged
@show result.x
@show result.solve_time_seconds
@show result.qp_time_seconds
@show result.iterations
