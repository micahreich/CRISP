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

problem = CRISP.make_crisp_problem_from_marble_data(data, "CRISPJuliaQPCC", "model", true)
params = CRISP.SolverParameters()
CRISP.set_hyper_parameter!(params, "trustRegionTol", 1e-4)
CRISP.set_hyper_parameter!(params, "trailTol", 1e-4)
CRISP.set_hyper_parameter!(params, "constraintTol", 1e-6)
CRISP.set_hyper_parameter!(params, "WeightedMode", 0.0)
CRISP.set_hyper_parameter!(params, "verbose", 0.0)

solver = CRISP.SolverInterface(problem, params)
CRISP.initialize!(solver, zeros(4))
converged = CRISP.solve!(solver)

@show converged
@show CRISP.has_converged(solver)
@show CRISP.get_solution_silent(solver)
@show CRISP.get_solve_time_seconds(solver)
@show CRISP.get_qp_solve_time_seconds(solver)
@show CRISP.get_iteration_count(solver)
