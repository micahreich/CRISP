#include <jlcxx/jlcxx.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "problem_core/MarbleProblemAdapter.h"
#include "solver_core/SolverInterface.h"

using namespace CRISP;

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

static void apply_solver_options(SolverParameters& params, jlcxx::ArrayRef<double, 1> option_values) {
    static constexpr std::array<const char*, 14> option_names = {
        "maxIterations",
        "trustRegionInitRadius",
        "trustRegionMaxRadius",
        "mu",
        "muMax",
        "etaLow",
        "etaHigh",
        "trailTol",
        "trustRegionTol",
        "constraintTol",
        "verbose",
        "WeightedMode",
        "WeightedTolFactor",
        "secondOrderCorrection",
    };

    if (option_values.size() != static_cast<long>(option_names.size())) {
        throw std::runtime_error("CRISP option vector has unexpected length");
    }

    for (long i = 0; i < option_values.size(); ++i) {
        const double value = option_values[i];
        if (!std::isnan(value)) {
            params.setParameters(option_names[static_cast<size_t>(i)], vector_t::Constant(1, value));
        }
    }
}

static jlcxx::Array<double> solve_qpcc_with_crisp(jlcxx::ArrayRef<double, 2> Q,
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
                                                  const std::string& problem_name,
                                                  const std::string& folder_name,
                                                  bool regenerate_library,
                                                  jlcxx::ArrayRef<double, 1> option_values) {
    MarbleProblemData data = make_data(Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r);
    vector_t initial = to_eigen(x0);
    if (initial.size() != data.q.size()) {
        throw std::runtime_error("Initial guess length does not match problem dimension");
    }

    SolverParameters params;
    apply_solver_options(params, option_values);

    auto problem = makeCrispProblemFromMarbleData(data, problem_name, folder_name, regenerate_library);
    SolverInterface solver(*problem, params);
    solver.initialize(initial);
    solver.solve();
    vector_t solution = solver.getSolutionSilent();

    jlcxx::Array<double> result(solution.size() + 4);
#if JULIA_VERSION_MAJOR > 1 || (JULIA_VERSION_MAJOR == 1 && JULIA_VERSION_MINOR >= 11)
    double* result_data = jl_array_data(result.wrapped(), double);
#else
    double* result_data = reinterpret_cast<double*>(jl_array_data(result.wrapped()));
#endif
    result_data[0] = solver.hasConverged() ? 1.0 : 0.0;
    result_data[1] = solver.getSolveTimeSeconds();
    result_data[2] = solver.getQpSolveTimeSeconds();
    result_data[3] = static_cast<double>(solver.getIterationCount());
    std::copy(solution.data(), solution.data() + solution.size(), result_data + 4);
    return result;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.method("_solve_qpcc_with_crisp", solve_qpcc_with_crisp);
}
