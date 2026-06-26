using CRISP
using LinearAlgebra

Q = 2.0 * Matrix(I, 4, 4)
q = zeros(4)
c0 = 0.0

result = solve_qpcc(Q, q, c0;
    J_eq = [1.0 0.0 0.0 0.0],
    b_eq = [-1.0],
    J_ineq = [0.0 1.0 0.0 0.0],
    b_ineq = [-1.0],
    L = [0.0 0.0 1.0 0.0],
    l = [1.0],
    R = [0.0 0.0 0.0 1.0],
    r = [-1.0],
)

@show result.x
@show result.solve_time_seconds
@show result.qp_time_seconds
@show result.iterations
