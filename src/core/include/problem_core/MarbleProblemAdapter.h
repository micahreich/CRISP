#ifndef MARBLE_PROBLEM_ADAPTER_H
#define MARBLE_PROBLEM_ADAPTER_H

#include "problem_core/OptimizationProblem.h"

#include <cstdint>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

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

inline void readExact(std::istream& in, char* data, std::streamsize bytes, const std::string& what) {
    if (!in.read(data, bytes)) {
        throw std::runtime_error("Truncated Marble problem while reading " + what);
    }
}

template <typename T>
inline T readScalar(std::istream& in, const std::string& what) {
    T value;
    readExact(in, reinterpret_cast<char*>(&value), static_cast<std::streamsize>(sizeof(T)), what);
    return value;
}

inline sparse_matrix_t readSparse(std::istream& in, const std::string& name) {
    const auto rows64 = readScalar<std::int64_t>(in, name + ".rows");
    const auto cols64 = readScalar<std::int64_t>(in, name + ".cols");
    const auto nnz64 = readScalar<std::int64_t>(in, name + ".nnz");
    if (rows64 < 0 || cols64 < 0 || nnz64 < 0) {
        throw std::runtime_error("Negative dimension in sparse block " + name);
    }

    std::vector<std::int64_t> colptr(static_cast<size_t>(cols64) + 1);
    std::vector<std::int64_t> rowval(static_cast<size_t>(nnz64));
    std::vector<scalar_t> nzval(static_cast<size_t>(nnz64));

    readExact(in, reinterpret_cast<char*>(colptr.data()),
              static_cast<std::streamsize>(colptr.size() * sizeof(std::int64_t)), name + ".colptr");
    readExact(in, reinterpret_cast<char*>(rowval.data()),
              static_cast<std::streamsize>(rowval.size() * sizeof(std::int64_t)), name + ".rowval");
    readExact(in, reinterpret_cast<char*>(nzval.data()),
              static_cast<std::streamsize>(nzval.size() * sizeof(scalar_t)), name + ".nzval");

    const int rows = static_cast<int>(rows64);
    const int cols = static_cast<int>(cols64);
    std::vector<Eigen::Triplet<scalar_t>> triplets;
    triplets.reserve(static_cast<size_t>(nnz64));
    for (int col = 0; col < cols; ++col) {
        const auto begin = colptr[static_cast<size_t>(col)];
        const auto end = colptr[static_cast<size_t>(col + 1)];
        if (begin < 0 || end < begin || end > nnz64) {
            throw std::runtime_error("Invalid CSC column pointer in sparse block " + name);
        }
        for (std::int64_t p = begin; p < end; ++p) {
            const auto row = rowval[static_cast<size_t>(p)];
            if (row < 0 || row >= rows64) {
                throw std::runtime_error("Invalid CSC row index in sparse block " + name);
            }
            triplets.emplace_back(static_cast<int>(row), col, nzval[static_cast<size_t>(p)]);
        }
    }

    sparse_matrix_t matrix(rows, cols);
    matrix.setFromTriplets(triplets.begin(), triplets.end());
    matrix.makeCompressed();
    return matrix;
}

inline vector_t readVector(std::istream& in, const std::string& name) {
    const auto len64 = readScalar<std::int64_t>(in, name + ".length");
    if (len64 < 0) {
        throw std::runtime_error("Negative length in vector block " + name);
    }

    vector_t vector(static_cast<int>(len64));
    readExact(in, reinterpret_cast<char*>(vector.data()),
              static_cast<std::streamsize>(static_cast<size_t>(len64) * sizeof(scalar_t)), name);
    return vector;
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

inline MarbleProblemData loadMarbleProblemData(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open Marble problem file: " + path);
    }

    const std::string expectedMagic = "MARBLEPB";
    std::vector<char> magic(expectedMagic.size());
    marble_adapter_detail::readExact(in, magic.data(), static_cast<std::streamsize>(magic.size()), "magic");
    if (std::string(magic.begin(), magic.end()) != expectedMagic) {
        throw std::runtime_error("Not a Marble problem file: " + path);
    }

    const auto version = marble_adapter_detail::readScalar<std::int64_t>(in, "version");
    if (version != 1) {
        throw std::runtime_error("Unsupported Marble problem version " + std::to_string(version));
    }

    MarbleProblemData data;
    data.c0 = marble_adapter_detail::readScalar<scalar_t>(in, "cost constant");
    data.Q = marble_adapter_detail::readSparse(in, "Q");
    data.q = marble_adapter_detail::readVector(in, "q");
    data.J_eq = marble_adapter_detail::readSparse(in, "J_eq");
    data.b_eq = marble_adapter_detail::readVector(in, "b_eq");
    data.J_ineq = marble_adapter_detail::readSparse(in, "J_ineq");
    data.b_ineq = marble_adapter_detail::readVector(in, "b_ineq");
    data.L = marble_adapter_detail::readSparse(in, "L");
    data.l = marble_adapter_detail::readVector(in, "l");
    data.R = marble_adapter_detail::readSparse(in, "R");
    data.r = marble_adapter_detail::readVector(in, "r");
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
    const std::string& path,
    const std::string& problemName = "MarbleDataProblem",
    const std::string& folderName = "model",
    bool regenerateLibrary = true) {

    return makeCrispProblemFromMarbleData(
        loadMarbleProblemData(path), problemName, folderName, regenerateLibrary);
}

} // namespace CRISP

#endif // MARBLE_PROBLEM_ADAPTER_H
