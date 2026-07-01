#include <jlcxx/jlcxx.hpp>

#include <algorithm>
#include <array>
#include <chrono>
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

// Result handed back to Julia. Exposed through per-field getters (registered in
// the module below) so the Julia side can assemble a clean NamedTuple rather
// than unpacking a flat, index-addressed array.
struct CrispResult {
    bool converged = false;
    double setup_time_seconds = 0.0;  // problem construction incl. CppAD code generation
    double solve_time_seconds = 0.0;  // solver loop only (excludes CppAD generation)
    double qp_time_seconds = 0.0;
    int iterations = 0;
    std::vector<double> x;
};

// Copy a std::vector into a freshly allocated Julia array.
static jlcxx::Array<double> to_julia(const std::vector<double>& v) {
    jlcxx::Array<double> out;
    for (double value : v)
        out.push_back(value);
    return out;
}

static CrispResult solve_qpcc_with_crisp(jlcxx::ArrayRef<double, 2> Q,
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

    // Time the problem construction separately: when regenerate_library is true
    // this is where CppAD generates and compiles the derivative library, which we
    // deliberately keep out of the reported solve time.
    const auto setup_start = std::chrono::high_resolution_clock::now();
    auto problem = makeCrispProblemFromMarbleData(data, problem_name, folder_name, regenerate_library);
    const auto setup_end = std::chrono::high_resolution_clock::now();

    SolverInterface solver(*problem, params);
    solver.initialize(initial);
    solver.solve();
    vector_t solution = solver.getSolutionSilent();

    CrispResult result;
    result.converged = solver.hasConverged();
    result.setup_time_seconds = std::chrono::duration<double>(setup_end - setup_start).count();
    result.solve_time_seconds = solver.getSolveTimeSeconds();
    result.qp_time_seconds = solver.getQpSolveTimeSeconds();
    result.iterations = solver.getIterationCount();
    result.x.assign(solution.data(), solution.data() + solution.size());
    return result;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<CrispResult>("CrispResultCxx")
        .method("converged", [](const CrispResult& r) { return r.converged; })
        .method("setup_time_seconds", [](const CrispResult& r) { return r.setup_time_seconds; })
        .method("solve_time_seconds", [](const CrispResult& r) { return r.solve_time_seconds; })
        .method("qp_time_seconds", [](const CrispResult& r) { return r.qp_time_seconds; })
        .method("iterations", [](const CrispResult& r) { return r.iterations; })
        .method("primal_solution", [](const CrispResult& r) { return to_julia(r.x); });

    mod.method("_solve_qpcc_with_crisp", solve_qpcc_with_crisp);
}
