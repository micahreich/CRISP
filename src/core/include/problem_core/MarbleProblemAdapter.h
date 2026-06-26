#ifndef MARBLE_PROBLEM_ADAPTER_H
#define MARBLE_PROBLEM_ADAPTER_H

#include "problem_core/OptimizationProblem.h"

#include <unsupported/Eigen/SparseExtra>

#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>

namespace CRISP {

struct MarbleProblemData {
    sparse_matrix_t Q;
    vector_t q;
    scalar_t c0 = 0.0;

    sparse_matrix_t J_eq;
    vector_t b_eq;

    sparse_matrix_t J_ineq;
    vector_t b_ineq;

    sparse_matrix_t L;
    vector_t l;

    sparse_matrix_t R;
    vector_t r;
};

namespace marble_adapter_detail {

inline std::string joinPath(const std::string& dir, const std::string& file) {
    if (dir.empty() || dir[dir.size() - 1] == '/') {
        return dir + file;
    }
    return dir + "/" + file;
}

inline sparse_matrix_t loadMarketSparse(const std::string& path, const std::string& name) {
    sparse_matrix_t matrix;
    if (!Eigen::loadMarket(matrix, path)) {
        throw std::runtime_error("Unable to load Matrix Market sparse block " + name + " from " + path);
    }
    matrix.makeCompressed();
    return matrix;
}

inline vector_t loadMarketVector(const std::string& path, const std::string& name) {
    vector_t vector;
    if (Eigen::loadMarketVector(vector, path)) {
        return vector;
    }

    sparse_matrix_t matrix;
    if (!Eigen::loadMarket(matrix, path)) {
        throw std::runtime_error("Unable to load Matrix Market vector block " + name + " from " + path);
    }
    if (matrix.cols() == 1) {
        return vector_t(matrix);
    }
    if (matrix.rows() == 1) {
        return vector_t(matrix.transpose());
    }
    throw std::runtime_error("Matrix Market vector block " + name + " must be a row or column vector");
}

inline scalar_t loadScalarText(const std::string& path, const std::string& name) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open scalar block " + name + " from " + path);
    }
    scalar_t value;
    if (!(in >> value)) {
        throw std::runtime_error("Unable to parse scalar block " + name + " from " + path);
    }
    return value;
}

inline ad_scalar_t sparseDotRow(const sparse_matrix_t& A, int row, const ad_vector_t& x) {
    ad_scalar_t value(0.0);
    for (sparse_matrix_t::InnerIterator it(A, row); it; ++it) {
        value += it.value() * x[it.col()];
    }
    return value;
}

inline void validateRows(const sparse_matrix_t& A, const vector_t& b, const std::string& name) {
    if (A.rows() != b.size()) {
        throw std::runtime_error(name + " row count does not match its offset vector length");
    }
}

} // namespace marble_adapter_detail

inline void validateMarbleProblemData(const MarbleProblemData& data) {
    const int n = data.q.size();
    if (data.Q.rows() != n || data.Q.cols() != n) {
        throw std::runtime_error("Marble Q must be square with size length(q)");
    }

    const sparse_matrix_t* blocks[] = {&data.J_eq, &data.J_ineq, &data.L, &data.R};
    const char* names[] = {"J_eq", "J_ineq", "L", "R"};
    for (int i = 0; i < 4; ++i) {
        if (blocks[i]->cols() != n) {
            throw std::runtime_error(std::string("Marble ") + names[i] + " must have length(q) columns");
        }
    }

    marble_adapter_detail::validateRows(data.J_eq, data.b_eq, "J_eq");
    marble_adapter_detail::validateRows(data.J_ineq, data.b_ineq, "J_ineq");
    marble_adapter_detail::validateRows(data.L, data.l, "L");
    marble_adapter_detail::validateRows(data.R, data.r, "R");
    if (data.L.rows() != data.R.rows()) {
        throw std::runtime_error("Marble L and R must have the same number of complementarity rows");
    }
}

inline MarbleProblemData loadMarbleProblemData(const std::string& dir) {
    MarbleProblemData data;
    data.c0 = marble_adapter_detail::loadScalarText(marble_adapter_detail::joinPath(dir, "c0.txt"), "c0");
    data.Q = marble_adapter_detail::loadMarketSparse(marble_adapter_detail::joinPath(dir, "Q_matrix.mtx"), "Q");
    data.q = marble_adapter_detail::loadMarketVector(marble_adapter_detail::joinPath(dir, "q_vector.mtx"), "q");
    data.J_eq = marble_adapter_detail::loadMarketSparse(marble_adapter_detail::joinPath(dir, "J_eq_matrix.mtx"), "J_eq");
    data.b_eq = marble_adapter_detail::loadMarketVector(marble_adapter_detail::joinPath(dir, "b_eq_vector.mtx"), "b_eq");
    data.J_ineq = marble_adapter_detail::loadMarketSparse(marble_adapter_detail::joinPath(dir, "J_ineq_matrix.mtx"), "J_ineq");
    data.b_ineq = marble_adapter_detail::loadMarketVector(marble_adapter_detail::joinPath(dir, "b_ineq_vector.mtx"), "b_ineq");
    data.L = marble_adapter_detail::loadMarketSparse(marble_adapter_detail::joinPath(dir, "L_matrix.mtx"), "L");
    data.l = marble_adapter_detail::loadMarketVector(marble_adapter_detail::joinPath(dir, "l_vector.mtx"), "l");
    data.R = marble_adapter_detail::loadMarketSparse(marble_adapter_detail::joinPath(dir, "R_matrix.mtx"), "R");
    data.r = marble_adapter_detail::loadMarketVector(marble_adapter_detail::joinPath(dir, "r_vector.mtx"), "r");
    validateMarbleProblemData(data);
    return data;
}

inline std::unique_ptr<OptimizationProblem> makeCrispProblemFromMarbleData(
    const MarbleProblemData& data,
    const std::string& problemName = "MarbleDataProblem",
    const std::string& folderName = "model",
    bool regenerateLibrary = true) {

    validateMarbleProblemData(data);
    const auto shared = std::make_shared<MarbleProblemData>(data);
    const size_t variableDim = static_cast<size_t>(shared->q.size());

    ad_function_t objective = [shared](const ad_vector_t& x, ad_vector_t& y) {
        y.resize(1);
        ad_scalar_t value(shared->c0);
        for (int row = 0; row < shared->Q.rows(); ++row) {
            for (sparse_matrix_t::InnerIterator it(shared->Q, row); it; ++it) {
                value += ad_scalar_t(0.5) * it.value() * x[it.row()] * x[it.col()];
            }
        }
        for (int i = 0; i < shared->q.size(); ++i) {
            value += shared->q[i] * x[i];
        }
        y[0] = value;
    };

    ad_function_t equality = [shared](const ad_vector_t& x, ad_vector_t& y) {
        y.resize(shared->J_eq.rows());
        for (int row = 0; row < shared->J_eq.rows(); ++row) {
            y[row] = marble_adapter_detail::sparseDotRow(shared->J_eq, row, x) + shared->b_eq[row];
        }
    };

    ad_function_t inequality = [shared](const ad_vector_t& x, ad_vector_t& y) {
        const int nJ = shared->J_ineq.rows();
        const int ncc = shared->L.rows();
        y.resize(nJ + 3 * ncc);

        for (int row = 0; row < nJ; ++row) {
            y[row] = marble_adapter_detail::sparseDotRow(shared->J_ineq, row, x) + shared->b_ineq[row];
        }

        for (int row = 0; row < ncc; ++row) {
            const ad_scalar_t left = marble_adapter_detail::sparseDotRow(shared->L, row, x) + shared->l[row];
            const ad_scalar_t right = marble_adapter_detail::sparseDotRow(shared->R, row, x) + shared->r[row];
            y[nJ + row] = left;
            y[nJ + ncc + row] = right;
            y[nJ + 2 * ncc + row] = -(left * right);
        }
    };

    auto problem = std::unique_ptr<OptimizationProblem>(new OptimizationProblem(variableDim, problemName));
    auto obj = std::make_shared<ObjectiveFunction>(
        variableDim, problemName, folderName, "marbleQuadraticObjective", objective, regenerateLibrary);

    problem->addObjective(obj);
    if (shared->J_eq.rows() > 0) {
        auto eq = std::make_shared<ConstraintFunction>(
            variableDim, problemName, folderName, "marbleEqualityConstraints", equality, regenerateLibrary);
        problem->addEqualityConstraint(eq);
    }
    if (shared->J_ineq.rows() + 3 * shared->L.rows() > 0) {
        auto ineq = std::make_shared<ConstraintFunction>(
            variableDim, problemName, folderName, "marbleInequalityComplementarityConstraints",
            inequality, regenerateLibrary);
        problem->addInequalityConstraint(ineq);
    }
    return problem;
}

inline std::unique_ptr<OptimizationProblem> makeCrispProblemFromMarbleFile(
    const std::string& dir,
    const std::string& problemName = "MarbleDataProblem",
    const std::string& folderName = "model",
    bool regenerateLibrary = true) {

    return makeCrispProblemFromMarbleData(
        loadMarbleProblemData(dir), problemName, folderName, regenerateLibrary);
}

} // namespace CRISP

#endif // MARBLE_PROBLEM_ADAPTER_H
