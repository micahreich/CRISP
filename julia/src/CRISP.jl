module CRISP

using CxxWrap
using LinearAlgebra
using Preferences

const _libcrisp_julia = @load_preference("libcrisp_julia_path",
    joinpath(@__DIR__, "..", "..", "build", "lib", "libcrisp_julia"))

@wrapmodule(() -> _libcrisp_julia)

function __init__()
    @initcxx
end

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

"""
    solve_qpcc(Q, q, c0=0.0; J_eq, b_eq, J_ineq, b_ineq, L, l, R, r, x0, ...)

Solve the matrix-form QPCC with CRISP:

    minimize    1/2 x'Qx + q'x + c0
    subject to  J_eq x + b_eq == 0
                J_ineq x + b_ineq >= 0
                0 <= Lx + l complements Rx + r >= 0

Returns `(x, solve_time_seconds, qp_time_seconds, iterations)`.
"""
function solve_qpcc(Q, q, c0::Real=0.0;
                    J_eq=nothing, b_eq=nothing,
                    J_ineq=nothing, b_ineq=nothing,
                    L=nothing, l=nothing,
                    R=nothing, r=nothing,
                    x0=nothing,
                    trust_region_tol=1e-5,
                    trail_tol=1e-5,
                    constraint_tol=1e-5,
                    weighted_mode=1.0,
                    verbose=0.0)
    qv = Vector{Float64}(q)
    n = length(qv)
    Qm = _mat(Q, n, n, "Q")

    Je = isnothing(J_eq) ? zeros(0, n) : Matrix{Float64}(J_eq)
    be = isnothing(b_eq) ? zeros(size(Je, 1)) : Vector{Float64}(b_eq)
    size(Je, 2) == n || throw(ArgumentError("J_eq must have $n columns"))
    length(be) == size(Je, 1) || throw(ArgumentError("b_eq length must match rows of J_eq"))

    Ji = isnothing(J_ineq) ? zeros(0, n) : Matrix{Float64}(J_ineq)
    bi = isnothing(b_ineq) ? zeros(size(Ji, 1)) : Vector{Float64}(b_ineq)
    size(Ji, 2) == n || throw(ArgumentError("J_ineq must have $n columns"))
    length(bi) == size(Ji, 1) || throw(ArgumentError("b_ineq length must match rows of J_ineq"))

    Lm = isnothing(L) ? zeros(0, n) : Matrix{Float64}(L)
    lv = isnothing(l) ? zeros(size(Lm, 1)) : Vector{Float64}(l)
    Rm = isnothing(R) ? zeros(size(Lm, 1), n) : Matrix{Float64}(R)
    rv = isnothing(r) ? zeros(size(Rm, 1)) : Vector{Float64}(r)
    size(Lm, 2) == n || throw(ArgumentError("L must have $n columns"))
    size(Rm, 2) == n || throw(ArgumentError("R must have $n columns"))
    size(Lm, 1) == size(Rm, 1) || throw(ArgumentError("L and R must have the same number of rows"))
    length(lv) == size(Lm, 1) || throw(ArgumentError("l length must match rows of L"))
    length(rv) == size(Rm, 1) || throw(ArgumentError("r length must match rows of R"))

    xinit = isnothing(x0) ? zeros(n) : _vec(x0, n, "x0")

    x, solve_time, qp_time, iterations = CRISP.solve_qpcc_dense(
        Qm, qv, Float64(c0), Je, be, Ji, bi, Lm, lv, Rm, rv, xinit,
        Float64(trust_region_tol), Float64(trail_tol), Float64(constraint_tol),
        Float64(weighted_mode), Float64(verbose))

    return (
        x = Vector{Float64}(x),
        solve_time_seconds = solve_time,
        qp_time_seconds = qp_time,
        iterations = iterations,
    )
end

export solve_qpcc

end
