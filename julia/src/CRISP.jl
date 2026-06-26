module CRISP

using CxxWrap
using Preferences

const _libcrisp_julia = @load_preference("libcrisp_julia_path",
    joinpath(@__DIR__, "..", "..", "build", "lib", "libcrisp_julia"))

@wrapmodule(() -> _libcrisp_julia)

function __init__()
    @initcxx
end

const OPTION_NAME_MAP = Dict{Symbol, String}(
    :max_iterations => "maxIterations",
    :trust_region_init_radius => "trustRegionInitRadius",
    :trust_region_max_radius => "trustRegionMaxRadius",
    :mu => "mu",
    :mu_max => "muMax",
    :eta_low => "etaLow",
    :eta_high => "etaHigh",
    :trail_tol => "trailTol",
    :trust_region_tol => "trustRegionTol",
    :constraint_tol => "constraintTol",
    :verbose => "verbose",
    :weighted_mode => "WeightedMode",
    :weighted_tol_factor => "WeightedTolFactor",
    :second_order_correction => "secondOrderCorrection",
)

_option_name(name::Symbol) = get(OPTION_NAME_MAP, name) do
    throw(ArgumentError("unknown CRISP option: $name"))
end
_option_name(name::AbstractString) = String(name)

set_hyper_parameter!(target::Union{SolverInterface, SolverParameters}, name, value::Real) =
    CRISP._set_hyper_parameter!(target, _option_name(name), Float64(value))

get_hyper_parameter(target::Union{SolverInterface, SolverParameters}, name) =
    CRISP._get_hyper_parameter(target, _option_name(name))

_mat(M, rows, cols, name) = begin
    A = Matrix{Float64}(M)
    size(A) == (rows, cols) || throw(ArgumentError("$name must have size ($rows, $cols), got $(size(A))"))
    A
end

_vec(v, len, name) = begin
    x = Vector{Float64}(v)
    length(x) == len || throw(ArgumentError("$name must have length $len, got $(length(x))"))
    x
end

function _qpcc_data(Q, q, c0::Real;
                    J_eq=nothing, c_eq=nothing,
                    J_ineq=nothing, c_ineq=nothing,
                    L=nothing, l=nothing,
                    R=nothing, r=nothing)
    qv = Vector{Float64}(q)
    n = length(qv)
    Qm = _mat(Q, n, n, "Q")

    Je = isnothing(J_eq) ? zeros(0, n) : Matrix{Float64}(J_eq)
    ce = isnothing(c_eq) ? zeros(size(Je, 1)) : Vector{Float64}(c_eq)
    size(Je, 2) == n || throw(ArgumentError("J_eq must have $n columns"))
    length(ce) == size(Je, 1) || throw(ArgumentError("c_eq length must match rows of J_eq"))

    Ji = isnothing(J_ineq) ? zeros(0, n) : Matrix{Float64}(J_ineq)
    ci = isnothing(c_ineq) ? zeros(size(Ji, 1)) : Vector{Float64}(c_ineq)
    size(Ji, 2) == n || throw(ArgumentError("J_ineq must have $n columns"))
    length(ci) == size(Ji, 1) || throw(ArgumentError("c_ineq length must match rows of J_ineq"))

    Lm = isnothing(L) ? zeros(0, n) : Matrix{Float64}(L)
    lv = isnothing(l) ? zeros(size(Lm, 1)) : Vector{Float64}(l)
    Rm = isnothing(R) ? zeros(size(Lm, 1), n) : Matrix{Float64}(R)
    rv = isnothing(r) ? zeros(size(Rm, 1)) : Vector{Float64}(r)
    size(Lm, 2) == n || throw(ArgumentError("L must have $n columns"))
    size(Rm, 2) == n || throw(ArgumentError("R must have $n columns"))
    size(Lm, 1) == size(Rm, 1) || throw(ArgumentError("L and R must have the same number of rows"))
    length(lv) == size(Lm, 1) || throw(ArgumentError("l length must match rows of L"))
    length(rv) == size(Rm, 1) || throw(ArgumentError("r length must match rows of R"))

    return (
        Q = Qm, q = qv, c0 = Float64(c0),
        J_eq = Je, c_eq = ce,
        J_ineq = Ji, c_ineq = ci,
        L = Lm, l = lv,
        R = Rm, r = rv,
    )
end

function make_crisp_problem_from_marble_data(data::NamedTuple,
                                             problem_name::AbstractString="CRISPJuliaQPCC",
                                             folder_name::AbstractString="model",
                                             regenerate_library::Bool=true)
    data = _qpcc_data(data.Q, data.q, data.c0;
        J_eq = data.J_eq, c_eq = data.c_eq,
        J_ineq = data.J_ineq, c_ineq = data.c_ineq,
        L = data.L, l = data.l,
        R = data.R, r = data.r)
    return CRISP.OptimizationProblem(
        data.Q, data.q, data.c0,
        data.J_eq, data.c_eq,
        data.J_ineq, data.c_ineq,
        data.L, data.l,
        data.R, data.r,
        String(problem_name), String(folder_name), regenerate_library)
end

function make_crisp_problem_from_marble_data(Q, q, c0::Real=0.0;
                                             J_eq=nothing, c_eq=nothing,
                                             J_ineq=nothing, c_ineq=nothing,
                                             L=nothing, l=nothing,
                                             R=nothing, r=nothing,
                                             problem_name::AbstractString="CRISPJuliaQPCC",
                                             folder_name::AbstractString="model",
                                             regenerate_library::Bool=true)
    data = _qpcc_data(Q, q, c0;
        J_eq = J_eq, c_eq = c_eq,
        J_ineq = J_ineq, c_ineq = c_ineq,
        L = L, l = l,
        R = R, r = r)
    return make_crisp_problem_from_marble_data(data, problem_name, folder_name, regenerate_library)
end

"""
    solve_qpcc_with_crisp(Q, q, c0=0.0; J_eq, c_eq, J_ineq, c_ineq, L, l, R, r, x0, ...)

Solve a matrix-form QPCC with CRISP:

    minimize    1/2 x'Qx + q'x + c0
    subject to  J_eq x + c_eq == 0
                J_ineq x + c_ineq >= 0
                0 <= Lx + l complements Rx + r >= 0

Extra keyword arguments are treated as CRISP solver options.
"""
function solve_qpcc_with_crisp(Q, q, c0::Real=0.0;
                               J_eq=nothing, c_eq=nothing,
                               J_ineq=nothing, c_ineq=nothing,
                               L=nothing, l=nothing,
                               R=nothing, r=nothing,
                               x0=nothing,
                               problem_name::AbstractString="CRISPJuliaQPCC",
                               folder_name::AbstractString="model",
                               regenerate_library::Bool=true,
                               kwargs...)
    data = _qpcc_data(Q, q, c0;
        J_eq = J_eq, c_eq = c_eq,
        J_ineq = J_ineq, c_ineq = c_ineq,
        L = L, l = l,
        R = R, r = r)
    return solve_qpcc_with_crisp(data;
        x0 = x0,
        problem_name = problem_name,
        folder_name = folder_name,
        regenerate_library = regenerate_library,
        kwargs...)
end

function solve_qpcc_with_crisp(data::NamedTuple;
                               x0=nothing,
                               problem_name::AbstractString="CRISPJuliaQPCC",
                               folder_name::AbstractString="model",
                               regenerate_library::Bool=true,
                               kwargs...)
    data = _qpcc_data(data.Q, data.q, data.c0;
        J_eq = data.J_eq, c_eq = data.c_eq,
        J_ineq = data.J_ineq, c_ineq = data.c_ineq,
        L = data.L, l = data.l,
        R = data.R, r = data.r)
    initial = isnothing(x0) ? zeros(length(data.q)) : _vec(x0, length(data.q), "x0")
    problem = make_crisp_problem_from_marble_data(data, problem_name, folder_name, regenerate_library)
    params = CRISP.SolverParameters()
    for (name, value) in kwargs
        set_hyper_parameter!(params, name, value)
    end
    solver = CRISP.SolverInterface(problem, params)
    CRISP.initialize!(solver, initial)
    converged = CRISP.solve!(solver)

    return (
        converged = converged,
        x = Vector{Float64}(CRISP.get_solution_silent(solver)),
        solve_time_seconds = CRISP.get_solve_time_seconds(solver),
        qp_time_seconds = CRISP.get_qp_solve_time_seconds(solver),
        iterations = CRISP.get_iteration_count(solver),
    )
end

end
