#include <jlcxx/jlcxx.hpp>

#include <algorithm>
#include <exception>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "problem_core/MarbleProblemAdapter.h"
#include "solver_core/SolverInterface.h"

using namespace CRISP;

template <typename F>
auto julia_call(F&& f) -> decltype(f()) {
    try {
        return f();
    } catch (const std::exception& e) {
        jl_error(e.what());
    } catch (...) {
        jl_error("Unknown C++ exception in CRISP");
    }
    throw std::runtime_error("unreachable after jl_error");
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> to_eigen(jlcxx::ArrayRef<T, 2>& arr) {
    const long rows = jl_array_dim(arr.wrapped(), 0);
    const long cols = jl_array_dim(arr.wrapped(), 1);
    if (rows == 0 || cols == 0) {
        return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>(rows, cols);
    }
    return Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Map(arr.data(), rows, cols);
}

template <typename T>
Eigen::Matrix<T, Eigen::Dynamic, 1> to_eigen(jlcxx::ArrayRef<T, 1>& arr) {
    if (arr.size() == 0) {
        return Eigen::Matrix<T, Eigen::Dynamic, 1>(0);
    }
    return Eigen::Matrix<T, Eigen::Dynamic, 1>::Map(arr.data(), arr.size());
}

template <typename T>
jlcxx::Array<T> make_julia_owned(Eigen::Matrix<T, Eigen::Dynamic, 1> vec) {
    jlcxx::Array<T> arr(vec.size());
#if JULIA_VERSION_MAJOR > 1 || (JULIA_VERSION_MAJOR == 1 && JULIA_VERSION_MINOR >= 11)
    T* dest_ptr = jl_array_data(arr.wrapped(), T);
#else
    T* dest_ptr = reinterpret_cast<T*>(jl_array_data(arr.wrapped()));
#endif
    std::copy(vec.data(), vec.data() + vec.size(), dest_ptr);
    return arr;
}

static sparse_matrix_t dense_to_sparse(const matrix_t& dense) {
    std::vector<Eigen::Triplet<scalar_t>> triplets;
    triplets.reserve(static_cast<size_t>(dense.rows()) * static_cast<size_t>(dense.cols()));
    for (int col = 0; col < dense.cols(); ++col) {
        for (int row = 0; row < dense.rows(); ++row) {
            const scalar_t value = dense(row, col);
            if (value != 0.0) {
                triplets.emplace_back(row, col, value);
            }
        }
    }

    sparse_matrix_t sparse(dense.rows(), dense.cols());
    sparse.setFromTriplets(triplets.begin(), triplets.end());
    sparse.makeCompressed();
    return sparse;
}

static MarbleProblemData make_data(jlcxx::ArrayRef<double, 2> Q,
                                   jlcxx::ArrayRef<double, 1> q,
                                   double c0,
                                   jlcxx::ArrayRef<double, 2> J_eq,
                                   jlcxx::ArrayRef<double, 1> b_eq,
                                   jlcxx::ArrayRef<double, 2> J_ineq,
                                   jlcxx::ArrayRef<double, 1> b_ineq,
                                   jlcxx::ArrayRef<double, 2> L,
                                   jlcxx::ArrayRef<double, 1> l,
                                   jlcxx::ArrayRef<double, 2> R,
                                   jlcxx::ArrayRef<double, 1> r) {
    MarbleProblemData data;
    data.Q = dense_to_sparse(to_eigen(Q));
    data.q = to_eigen(q);
    data.c0 = c0;
    data.J_eq = dense_to_sparse(to_eigen(J_eq));
    data.b_eq = to_eigen(b_eq);
    data.J_ineq = dense_to_sparse(to_eigen(J_ineq));
    data.b_ineq = to_eigen(b_ineq);
    data.L = dense_to_sparse(to_eigen(L));
    data.l = to_eigen(l);
    data.R = dense_to_sparse(to_eigen(R));
    data.r = to_eigen(r);
    validateMarbleProblemData(data);
    return data;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.method("solve_qpcc_dense", [](jlcxx::ArrayRef<double, 2> Q,
                                      jlcxx::ArrayRef<double, 1> q,
                                      double c0,
                                      jlcxx::ArrayRef<double, 2> J_eq,
                                      jlcxx::ArrayRef<double, 1> b_eq,
                                      jlcxx::ArrayRef<double, 2> J_ineq,
                                      jlcxx::ArrayRef<double, 1> b_ineq,
                                      jlcxx::ArrayRef<double, 2> L,
                                      jlcxx::ArrayRef<double, 1> l,
                                      jlcxx::ArrayRef<double, 2> R,
                                      jlcxx::ArrayRef<double, 1> r,
                                      jlcxx::ArrayRef<double, 1> x0,
                                      double trust_region_tol,
                                      double trail_tol,
                                      double constraint_tol,
                                      double weighted_mode,
                                      double verbose) {
        return julia_call([&]() {
            MarbleProblemData data = make_data(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r);
            vector_t initial = to_eigen(x0);
            if (initial.size() != data.q.size()) {
                throw std::runtime_error("Initial guess length does not match problem dimension");
            }

            auto problem = makeCrispProblemFromMarbleData(data, "JuliaMarbleDataProblem", "model", true);
            SolverParameters params;
            SolverInterface solver(*problem, params);
            solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, trust_region_tol));
            solver.setHyperParameters("trailTol", vector_t::Constant(1, trail_tol));
            solver.setHyperParameters("constraintTol", vector_t::Constant(1, constraint_tol));
            solver.setHyperParameters("WeightedMode", vector_t::Constant(1, weighted_mode));
            solver.setHyperParameters("verbose", vector_t::Constant(1, verbose));

            solver.initialize(initial);
            solver.solve();
            vector_t solution = solver.getSolutionSilent();
            return std::make_tuple(
                make_julia_owned<double>(std::move(solution)),
                solver.getSolveTimeSeconds(),
                solver.getQpSolveTimeSeconds(),
                static_cast<int64_t>(solver.getIterationCount()));
        });
    });
}
