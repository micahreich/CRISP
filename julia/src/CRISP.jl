module CRISP

using CxxWrap
using Preferences

const _libcrisp_julia = @load_preference("libcrisp_julia_path",
    joinpath(@__DIR__, "..", "..", "build", "lib", "libcrisp_julia"))

const DEFAULT_MODEL_FOLDER = joinpath(@__DIR__, "..", "..", "model")

@wrapmodule(() -> _libcrisp_julia)

function __init__()
    @initcxx
end

const OPTION_NAMES = (
    :max_iterations,
    :trust_region_init_radius,
    :trust_region_max_radius,
    :mu,
    :mu_max,
    :eta_low,
    :eta_high,
    :trail_tol,
    :trust_region_tol,
    :constraint_tol,
    :verbose,
    :weighted_mode,
    :weighted_tol_factor,
    :second_order_correction,
)

const OPTION_INDEX = Dict(name => i for (i, name) in pairs(OPTION_NAMES))

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

function _qpcc_data(data::NamedTuple)
    q = Vector{Float64}(data.q)
    n = length(q)
    Q = _mat(data.Q, n, n, "Q")

    J_eq = Matrix{Float64}(data.J_eq)
    b_eq = Vector{Float64}(data.b_eq)
    size(J_eq, 2) == n || throw(ArgumentError("J_eq must have $n columns"))
    length(b_eq) == size(J_eq, 1) || throw(ArgumentError("b_eq length must match rows of J_eq"))

    J_ineq = Matrix{Float64}(data.J_ineq)
    b_ineq = Vector{Float64}(data.b_ineq)
    size(J_ineq, 2) == n || throw(ArgumentError("J_ineq must have $n columns"))
    length(b_ineq) == size(J_ineq, 1) || throw(ArgumentError("b_ineq length must match rows of J_ineq"))

    L = Matrix{Float64}(data.L)
    l = Vector{Float64}(data.l)
    R = Matrix{Float64}(data.R)
    r = Vector{Float64}(data.r)
    size(L, 2) == n || throw(ArgumentError("L must have $n columns"))
    size(R, 2) == n || throw(ArgumentError("R must have $n columns"))
    size(L, 1) == size(R, 1) || throw(ArgumentError("L and R must have the same number of rows"))
    length(l) == size(L, 1) || throw(ArgumentError("l length must match rows of L"))
    length(r) == size(R, 1) || throw(ArgumentError("r length must match rows of R"))

    return (
        Q = Q, q = q, c0 = Float64(data.c0),
        J_eq = J_eq, b_eq = b_eq,
        J_ineq = J_ineq, b_ineq = b_ineq,
        L = L, l = l,
        R = R, r = r,
    )
end

function _option_values(kwargs)
    values = fill(NaN, length(OPTION_NAMES))
    for (name, value) in pairs(kwargs)
        index = get(OPTION_INDEX, name, nothing)
        isnothing(index) && throw(ArgumentError("unknown CRISP option: $name"))
        values[index] = Float64(value)
    end
    return values
end

"""
    solve_qpcc_with_crisp(data; x0, problem_name, folder_name, regenerate_library, kwargs...)

Solve Marble-form QPCC data with CRISP:

    minimize    1/2 x'Qx + q'x + c0
    subject to  J_eq x + b_eq == 0
                J_ineq x + b_ineq >= 0
                0 <= Lx + l complements Rx + r >= 0

`data` must be a named tuple with fields `Q`, `q`, `c0`, `J_eq`, `b_eq`,
`J_ineq`, `b_ineq`, `L`, `l`, `R`, and `r`.

Solver hyperparameters are accepted as snake-case keyword arguments, for
example `trust_region_tol`, `trail_tol`, `constraint_tol`, `weighted_mode`,
and `verbose`.

Returns a NamedTuple `(; converged, x, setup_time_seconds, solve_time_seconds,
qp_time_seconds, iterations)`. `setup_time_seconds` covers problem construction,
including CppAD code generation/compilation when `regenerate_library` is `true`;
`solve_time_seconds` measures only the solver loop and excludes that generation.
"""
function solve_qpcc_with_crisp(data::NamedTuple;
                               x0=nothing,
                               problem_name::AbstractString="CRISPJuliaQPCC",
                               folder_name::AbstractString=DEFAULT_MODEL_FOLDER,
                               regenerate_library::Bool=true,
                               kwargs...)
    data = _qpcc_data(data)
    initial = isnothing(x0) ? zeros(length(data.q)) : _vec(x0, length(data.q), "x0")
    options = _option_values(kwargs)

    result = CRISP._solve_qpcc_with_crisp(
        data.Q, data.q, data.c0,
        data.J_eq, data.b_eq,
        data.J_ineq, data.b_ineq,
        data.L, data.l,
        data.R, data.r,
        initial,
        String(problem_name), String(folder_name), regenerate_library,
        options,
    )

    return (
        converged = converged(result),
        x = Vector{Float64}(primal_solution(result)),
        setup_time_seconds = setup_time_seconds(result),
        solve_time_seconds = solve_time_seconds(result),
        qp_time_seconds = qp_time_seconds(result),
        iterations = Int(iterations(result)),
    )
end

end
