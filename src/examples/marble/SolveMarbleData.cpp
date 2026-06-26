#include "problem_core/MarbleProblemAdapter.h"
#include "solver_core/SolverInterface.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace CRISP;

namespace {

vector_t loadVectorFromTextFile(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Unable to open initial guess file: " + path);
    }

    std::vector<scalar_t> values;
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream stream(line);
        scalar_t value;
        while (stream >> value) {
            values.push_back(value);
        }
    }

    vector_t x(static_cast<int>(values.size()));
    for (int i = 0; i < x.size(); ++i) {
        x[i] = values[static_cast<size_t>(i)];
    }
    return x;
}

void saveVectorToTextFile(const vector_t& x, const std::string& path) {
    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Unable to open solution output file: " + path);
    }
    out << std::setprecision(17);
    for (int i = 0; i < x.size(); ++i) {
        out << x[i] << '\n';
    }
}

scalar_t maxAbs(const vector_t& x) {
    return x.size() == 0 ? 0.0 : x.array().abs().maxCoeff();
}

scalar_t minValue(const vector_t& x) {
    return x.size() == 0 ? 0.0 : x.minCoeff();
}

scalar_t objectiveValue(const MarbleProblemData& data, const vector_t& x) {
    return 0.5 * x.dot(data.Q * x) + data.q.dot(x) + data.c0;
}

void printMarbleResiduals(const MarbleProblemData& data, const vector_t& x) {
    const vector_t eq = data.J_eq * x + data.b_eq;
    const vector_t ineq = data.J_ineq * x + data.b_ineq;
    const vector_t left = data.L * x + data.l;
    const vector_t right = data.R * x + data.r;
    const vector_t product = left.array() * right.array();

    std::cout << "Objective:                 " << objectiveValue(data, x) << '\n';
    std::cout << "max |J_eq*x + b_eq|:       " << maxAbs(eq) << '\n';
    std::cout << "min  J_ineq*x + b_ineq:    " << minValue(ineq) << '\n';
    std::cout << "min  L*x + l:              " << minValue(left) << '\n';
    std::cout << "min  R*x + r:              " << minValue(right) << '\n';
    std::cout << "max |(L*x+l).*(R*x+r)|:    " << maxAbs(product) << '\n';
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || argc > 4) {
        std::cerr << "Usage: " << argv[0] << " problem.marble [initial_guess.txt] [solution.txt]\n";
        return 1;
    }

    try {
        const std::string problemPath = argv[1];
        const MarbleProblemData data = loadMarbleProblemData(problemPath);
        auto problem = makeCrispProblemFromMarbleData(data, "MarbleDataProblem", "model", true);

        vector_t x0;
        if (argc >= 3) {
            x0 = loadVectorFromTextFile(argv[2]);
            if (x0.size() != data.q.size()) {
                throw std::runtime_error("Initial guess length does not match Marble problem dimension");
            }
        } else {
            x0 = vector_t::Zero(data.q.size());
        }

        SolverParameters params;
        SolverInterface solver(*problem, params);
        solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-5));
        solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-5));
        solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-5));
        solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1.0));

        std::cout << "Loaded Marble problem with " << data.q.size() << " variables, "
                  << data.J_eq.rows() << " equality rows, "
                  << data.J_ineq.rows() << " inequality rows, and "
                  << data.L.rows() << " complementarity pairs.\n";

        solver.initialize(x0);
        solver.solve();
        const vector_t x = solver.getSolution();
        std::cout << "Timing summary: solve " << solver.getSolveTimeSeconds()
                  << "s, QP " << solver.getQpSolveTimeSeconds()
                  << "s, iterations " << solver.getIterationCount() << '\n';

        std::cout << "\nResiduals in the original Marble form:\n";
        printMarbleResiduals(data, x);

        if (argc == 4) {
            saveVectorToTextFile(x, argv[3]);
            std::cout << "Saved solution to " << argv[3] << '\n';
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
