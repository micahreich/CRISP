# Run from the CRISP checkout with:
#   julia src/examples/marble/write_simple_problem.jl
#
# This writes the problem
#
#   minimize    x'x
#   subject to  x1 == 1
#               x2 >= 1
#               0 <= x3 + 1  perpendicular  x4 - 1 >= 0
#
# in the binary matrix format consumed by `marble_data_example`.

using LinearAlgebra
using SparseArrays

const PROBLEM_MAGIC = Vector{UInt8}(codeunits("MARBLEPB"))
const PROBLEM_VERSION = Int64(1)

function write_sparse(io::IO, A::AbstractMatrix)
    S = sparse(A)
    write(io, Int64(size(S, 1)), Int64(size(S, 2)), Int64(nnz(S)))
    write(io, Int64.(SparseArrays.getcolptr(S) .- 1))
    write(io, Int64.(SparseArrays.rowvals(S) .- 1))
    write(io, Float64.(SparseArrays.nonzeros(S)))
end

function write_vector(io::IO, v::AbstractVector)
    write(io, Int64(length(v)))
    write(io, Float64.(collect(v)))
end

function write_problem(path::AbstractString, data)
    open(path, "w") do io
        write(io, PROBLEM_MAGIC)
        write(io, PROBLEM_VERSION)
        write(io, Float64(data.c0))
        write_sparse(io, data.Q);      write_vector(io, data.q)
        write_sparse(io, data.J_eq);   write_vector(io, data.b_eq)
        write_sparse(io, data.J_ineq); write_vector(io, data.b_ineq)
        write_sparse(io, data.L);      write_vector(io, data.l)
        write_sparse(io, data.R);      write_vector(io, data.r)
    end
    return path
end

Q = 2.0 * Matrix(1.0I, 4, 4)
q = zeros(4)
c0 = 0.0

J_eq = [1.0 0.0 0.0 0.0]
b_eq = [-1.0]

J_ineq = [0.0 1.0 0.0 0.0]
b_ineq = [-1.0]

L = [0.0 0.0 1.0 0.0]
l = [1.0]

R = [0.0 0.0 0.0 1.0]
r = [-1.0]

data = (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
out = joinpath(@__DIR__, "simple_qpcc.marble")
write_problem(out, data)
println(out)
