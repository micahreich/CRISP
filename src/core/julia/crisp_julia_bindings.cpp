#include <jlcxx/jlcxx.hpp>

#include <algorithm>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "problem_core/MarbleProblemAdapter.h"
#include "solver_core/SolverInterface.h"

using namespace CRISP;

class JuliaOptimizationProblem {
public:
    explicit JuliaOptimizationProblem(std::unique_ptr<OptimizationProblem> problem)
        : problem_(std::move(problem)) {}

    OptimizationProblem& get() {
        return *problem_;
    }

    const OptimizationProblem& get() const {
        return *problem_;
    }

    std::shared_ptr<OptimizationProblem> shared() const {
        return problem_;
    }

private:
    std::shared_ptr<OptimizationProblem> problem_;
};

class JuliaSolverParameters {
public:
    SolverParameters& get() {
        return parameters_;
    }

    const SolverParameters& get() const {
        return parameters_;
    }

    void setHyperParameter(const std::string& name, double value) {
        parameters_.setParameters(name, vector_t::Constant(1, value));
    }

    double getHyperParameter(const std::string& name) const {
        return parameters_.getParameters(name)(0);
    }

private:
    SolverParameters parameters_;
};

class JuliaSolverInterface {
public:
    explicit JuliaSolverInterface(JuliaOptimizationProblem& problem)
        : problem_(problem.shared()),
          parameters_(),
          solver_(std::make_unique<SolverInterface>(*problem_, parameters_)) {}

    JuliaSolverInterface(JuliaOptimizationProblem& problem, JuliaSolverParameters& parameters)
        : problem_(problem.shared()),
          parameters_(parameters.get()),
          solver_(std::make_unique<SolverInterface>(*problem_, parameters_)) {}

    void setHyperParameter(const std::string& name, double value) {
        parameters_.setParameters(name, vector_t::Constant(1, value));
        solver_->setHyperParameters(name, vector_t::Constant(1, value));
    }

    double getHyperParameter(const std::string& name) const {
        return parameters_.getParameters(name)(0);
    }

    void initialize(const vector_t& initial) {
        if (initial.size() != static_cast<long>(problem_->getVariableDim())) {
            throw std::runtime_error("Initial guess length does not match problem dimension");
        }
        solver_->initialize(initial);
    }

    bool solve() {
        solver_->solve();
        return solver_->hasConverged();
    }

    vector_t getSolution() {
        return solver_->getSolutionSilent();
    }

    double getSolveTimeSeconds() const {
        return solver_->getSolveTimeSeconds();
    }

    double getQpSolveTimeSeconds() const {
        return solver_->getQpSolveTimeSeconds();
    }

    int64_t getIterationCount() const {
        return static_cast<int64_t>(solver_->getIterationCount());
    }

    bool hasConverged() const {
        return solver_->hasConverged();
    }

private:
    std::shared_ptr<OptimizationProblem> problem_;
    SolverParameters parameters_;
    std::unique_ptr<SolverInterface> solver_;
};

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
                                   jlcxx::ArrayRef<double, 1> c_eq,
                                   jlcxx::ArrayRef<double, 2> J_ineq,
                                   jlcxx::ArrayRef<double, 1> c_ineq,
                                   jlcxx::ArrayRef<double, 2> L,
                                   jlcxx::ArrayRef<double, 1> l,
                                   jlcxx::ArrayRef<double, 2> R,
                                   jlcxx::ArrayRef<double, 1> r) {
    MarbleProblemData data;
    data.Q = dense_to_sparse(to_eigen(Q));
    data.q = to_eigen(q);
    data.c0 = c0;
    data.J_eq = dense_to_sparse(to_eigen(J_eq));
    data.c_eq = to_eigen(c_eq);
    data.J_ineq = dense_to_sparse(to_eigen(J_ineq));
    data.c_ineq = to_eigen(c_ineq);
    data.L = dense_to_sparse(to_eigen(L));
    data.l = to_eigen(l);
    data.R = dense_to_sparse(to_eigen(R));
    data.r = to_eigen(r);
    validateMarbleProblemData(data);
    return data;
}

JLCXX_MODULE define_julia_module(jlcxx::Module& mod) {
    mod.add_type<JuliaOptimizationProblem>("OptimizationProblem")
        .constructor([](jlcxx::ArrayRef<double, 2> Q,
                        jlcxx::ArrayRef<double, 1> q,
                        double c0,
                        jlcxx::ArrayRef<double, 2> J_eq,
                        jlcxx::ArrayRef<double, 1> c_eq,
                        jlcxx::ArrayRef<double, 2> J_ineq,
                        jlcxx::ArrayRef<double, 1> c_ineq,
                        jlcxx::ArrayRef<double, 2> L,
                        jlcxx::ArrayRef<double, 1> l,
                        jlcxx::ArrayRef<double, 2> R,
                        jlcxx::ArrayRef<double, 1> r,
                        const std::string& problem_name,
                        const std::string& folder_name,
                        bool regenerate_library) {
            MarbleProblemData data = make_data(Q, q, c0, J_eq, c_eq, J_ineq, c_ineq, L, l, R, r);
            return new JuliaOptimizationProblem(
                makeCrispProblemFromMarbleData(data, problem_name, folder_name, regenerate_library));
        })
        .method("get_variable_dim", [](const JuliaOptimizationProblem& p) {
            return static_cast<int64_t>(p.get().getVariableDim());
        })
        .method("get_num_objectives", [](const JuliaOptimizationProblem& p) {
            return static_cast<int64_t>(p.get().getNumObjectives());
        })
        .method("get_num_equality_constraints", [](const JuliaOptimizationProblem& p) {
            return static_cast<int64_t>(p.get().getNumEqualityConstraints());
        })
        .method("get_num_inequality_constraints", [](const JuliaOptimizationProblem& p) {
            return static_cast<int64_t>(p.get().getNumInequalityConstraints());
        })
        .method("get_problem_name", [](const JuliaOptimizationProblem& p) {
            return p.get().getProblemName();
        });

    mod.add_type<JuliaSolverParameters>("SolverParameters")
        .constructor()
        .method("_set_hyper_parameter!", [](JuliaSolverParameters& p, const std::string& name, double value) {
            p.setHyperParameter(name, value);
        })
        .method("_get_hyper_parameter", [](const JuliaSolverParameters& p, const std::string& name) {
            return p.getHyperParameter(name);
        });

    mod.add_type<JuliaSolverInterface>("SolverInterface")
        .constructor<JuliaOptimizationProblem&>()
        .constructor<JuliaOptimizationProblem&, JuliaSolverParameters&>()
        .method("_set_hyper_parameter!", [](JuliaSolverInterface& s, const std::string& name, double value) {
            s.setHyperParameter(name, value);
        })
        .method("_get_hyper_parameter", [](const JuliaSolverInterface& s, const std::string& name) {
            return s.getHyperParameter(name);
        })
        .method("initialize!", [](JuliaSolverInterface& s, jlcxx::ArrayRef<double, 1> x0) {
            s.initialize(to_eigen(x0));
        })
        .method("solve!", [](JuliaSolverInterface& s) {
            return s.solve();
        })
        .method("get_solution_silent", [](JuliaSolverInterface& s) {
            return make_julia_owned<double>(s.getSolution());
        })
        .method("get_solve_time_seconds", [](const JuliaSolverInterface& s) {
            return s.getSolveTimeSeconds();
        })
        .method("get_qp_solve_time_seconds", [](const JuliaSolverInterface& s) {
            return s.getQpSolveTimeSeconds();
        })
        .method("get_iteration_count", [](const JuliaSolverInterface& s) {
            return s.getIterationCount();
        })
        .method("has_converged", [](const JuliaSolverInterface& s) {
            return s.hasConverged();
        });

}
