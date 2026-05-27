#include "solver_core/SolverInterface.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include "math.h"

using namespace CRISP;

// Define model parameters for pushT
const scalar_t l = 0.05;
const scalar_t m = 1;
const scalar_t mu = 0.4;
const scalar_t g = 9.8;
const scalar_t r = 2.8 * l;
const scalar_t c = 0.4;
const scalar_t dc = 2.6429;
const scalar_t dt = 0.05;
const size_t N = 50; // number of time steps
const size_t num_state = 19;
const size_t num_control = 10;

void saveTrajectoryToTextFile(const Eigen::VectorXd& x, size_t steps, size_t stepWidth, const std::string& fileName) {
    if (x.size() != static_cast<int>(steps * stepWidth)) {
        throw std::runtime_error("Trajectory size does not match steps * stepWidth.");
    }

    std::ofstream outFile(fileName);
    if (!outFile.is_open()) {
        throw std::runtime_error("Unable to open file for writing: " + fileName);
    }

    outFile.precision(17);
    for (size_t i = 0; i < steps; ++i) {
        for (size_t j = 0; j < stepWidth; ++j) {
            outFile << x[i * stepWidth + j];
            if (j + 1 < stepWidth) {
                outFile << " ";
            }
        }
        outFile << "\n";
    }
}

// define the dynamics constraints
ad_function_t pushTDynamicConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 12 + 9);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        // Extract state and control for current and next time steps
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];
        ad_scalar_t c_theta = x[idx + 27];
        ad_scalar_t s_theta = x[idx + 28];

        if (i < N-1 ){
        ad_scalar_t px_next = x[idx + (num_state + num_control) + 0];
        ad_scalar_t py_next = x[idx + (num_state + num_control) + 1];
        ad_scalar_t theta_next = x[idx + (num_state + num_control) + 2];

        ad_scalar_t px_dot = (1/(mu*m*g))*(cos(theta_i)*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) - sin(theta_i)*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));
        ad_scalar_t py_dot = (1/(mu*m*g))*(sin(theta_i)*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) + cos(theta_i)*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));
        ad_scalar_t theta_dot = (1/(mu*m*g*c*r))*(-cy_i*(lambda2_i + lambda4_i + lambda6_i + lambda8_i) + cx_i*(lambda1_i + lambda3_i + lambda5_i + lambda7_i));

        // Explicit State Update
        y.segment(i * 12, 12) << px_next - px_i - px_dot * dt,
                                py_next - py_i - py_dot * dt,
                                theta_next - theta_i - theta_dot * dt,
                                (cx_i - 2*l) - v1_i + w1_i,
                                (cy_i - (4-dc)*l) - v2_i + w2_i,
                                (cy_i - (3-dc)*l) - v3_i + w3_i,
                                (cx_i - 0.5*l) - v4_i + w4_i,
                                (cy_i + dc*l) - v5_i + w5_i,
                                (cx_i + 0.5*l) - v6_i + w6_i,
                                (cx_i + 2*l) - v7_i + w7_i,
                                c_theta - cos(theta_i),
                                s_theta - sin(theta_i);
        }
        else{
            y.segment(i * 12, 9) << cx_i - 2*l - v1_i + w1_i,
                                    cy_i - (4-dc)*l - v2_i + w2_i,
                                    (cy_i - (3-dc)*l) - v3_i + w3_i,
                                    (cx_i - 0.5*l) - v4_i + w4_i,
                                    (cy_i + dc*l) - v5_i + w5_i,
                                    (cx_i + 0.5*l) - v6_i + w6_i,
                                    (cx_i + 2*l) - v7_i + w7_i,
                                    c_theta - cos(theta_i),
                                    s_theta - sin(theta_i);
        }
    }
};

// contact implicit constraints for pushT
ad_function_t pushTContactConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize(N * 41);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];

        y.segment(i * 41, 41) << v1_i,
                            w1_i,
                            -v1_i * w1_i,
                            v2_i,
                            w2_i,
                            -v2_i * w2_i,
                            v3_i,
                            w3_i,
                            -v3_i * w3_i,
                            v4_i,
                            w4_i,
                            -v4_i * w4_i,
                            v5_i,
                            w5_i,
                            -v5_i * w5_i,
                            v6_i,
                            w6_i,
                            -v6_i * w6_i,
                            v7_i,
                            w7_i,
                            -v7_i * w7_i,
                            cx_i + 2*l,
                            2*l - cx_i,
                            cy_i + dc * l,
                            (4-dc)*l - cy_i,
                            -lambda1_i,
                            -lambda2_i,
                             lambda3_i,
                            -lambda4_i,
                            lambda5_i,
                            lambda6_i,
                            lambda7_i,
                            lambda8_i,
                            -(-lambda1_i)*((4-dc)*l - cy_i),
                            -(-lambda2_i)*(v1_i + w1_i + v2_i + w2_i + v3_i + w3_i- l),
                            -(lambda3_i)*(v1_i + w1_i + v3_i + w3_i + v4_i + w4_i - 1.5*l),
                            -(-lambda4_i)*(v3_i + w3_i + v4_i + w4_i + v5_i + w5_i - 3.0*l),
                            -(lambda5_i)*(v4_i + w4_i + v5_i + w5_i + v6_i + w6_i - l),
                            -(lambda6_i)*(v3_i + w3_i + v5_i + w5_i + v6_i + w6_i - 3.0*l),
                            -(lambda7_i)*(v3_i + w3_i + v6_i + w6_i + v7_i + w7_i - 1.5*l),
                            -(lambda8_i)*(v2_i + w2_i + v3_i + w3_i + v7_i + w7_i - l);
    }
};

// allow only one contact force at a time
ad_function_t pushTContactSingleForceConstraints = [](const ad_vector_t& x, ad_vector_t& y) {
    y.resize((N - 1) * 28);
    for (size_t i = 0; i < N - 1; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];

        y.segment(i * 28, 28) << -(-lambda1_i * (-lambda2_i)),
                            -(-lambda1_i * lambda3_i),
                            -(-lambda1_i * (-lambda4_i)),
                            -(-lambda1_i * lambda5_i),
                            -(-lambda1_i * (lambda6_i)),
                            -(-lambda1_i * (lambda7_i)),
                            -(-lambda1_i * (lambda8_i)),
                            -(-lambda2_i * lambda3_i),
                            -(-lambda2_i * (-lambda4_i)),
                            -(-lambda2_i * lambda5_i),
                            -(-lambda2_i * (lambda6_i)),
                            -(-lambda2_i * (lambda7_i)),
                            -(-lambda2_i * (lambda8_i)),
                            -(lambda3_i * (-lambda4_i)),
                            -(lambda3_i * lambda5_i),
                            -(lambda3_i * (lambda6_i)),
                            -(lambda3_i * (lambda7_i)),
                            -(lambda3_i * (lambda8_i)),
                            -(-lambda4_i * lambda5_i),
                            -(-lambda4_i * (lambda6_i)),
                            -(-lambda4_i * (lambda7_i)),
                            -(-lambda4_i * (lambda8_i)),
                            -(lambda5_i * (lambda6_i)),
                            -(lambda5_i * (lambda7_i)),
                            -(lambda5_i * (lambda8_i)),
                            -(lambda6_i * (lambda7_i)),
                            -(lambda6_i * (lambda8_i)),
                            -(lambda7_i * (lambda8_i));

    }
};

// initial constraints
ad_function_with_param_t pushTInitialConstraints = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(4);
    y.segment(0, 4) << x[0] - p[0],
                    x[1] - p[1],
                    x[27] - p[2],
                    x[28] - p[3];
};

// cost function for pushT
ad_function_with_param_t pushTObjective = [](const ad_vector_t& x, const ad_vector_t& p, ad_vector_t& y) {
    y.resize(1);
    y[0] = 0.0;
    ad_scalar_t tracking_cost(0.0);
    ad_scalar_t control_cost(0.0);
    for (size_t i = 0; i < N; ++i) {
        size_t idx = i * (num_state + num_control);
        ad_scalar_t px_i = x[idx + 0];
        ad_scalar_t py_i = x[idx + 1];
        ad_scalar_t theta_i = x[idx + 2];
        ad_scalar_t cx_i = x[idx + 3];
        ad_scalar_t cy_i = x[idx + 4];
        ad_scalar_t v1_i = x[idx + 5];
        ad_scalar_t w1_i = x[idx + 6];
        ad_scalar_t v2_i = x[idx + 7];
        ad_scalar_t w2_i = x[idx + 8];
        ad_scalar_t v3_i = x[idx + 9];
        ad_scalar_t w3_i = x[idx + 10];
        ad_scalar_t v4_i = x[idx + 11];
        ad_scalar_t w4_i = x[idx + 12];
        ad_scalar_t v5_i = x[idx + 13];
        ad_scalar_t w5_i = x[idx + 14];
        ad_scalar_t v6_i = x[idx + 15];
        ad_scalar_t w6_i = x[idx + 16];
        ad_scalar_t v7_i = x[idx + 17];
        ad_scalar_t w7_i = x[idx + 18];
        ad_scalar_t lambda1_i = x[idx + 19];
        ad_scalar_t lambda2_i = x[idx + 20];
        ad_scalar_t lambda3_i = x[idx + 21];
        ad_scalar_t lambda4_i = x[idx + 22];
        ad_scalar_t lambda5_i = x[idx + 23];
        ad_scalar_t lambda6_i = x[idx + 24];
        ad_scalar_t lambda7_i = x[idx + 25];
        ad_scalar_t lambda8_i = x[idx + 26];
        ad_scalar_t c_theta = x[idx + 27];
        ad_scalar_t s_theta = x[idx + 28];
        ad_matrix_t Q(4, 4);
        ad_matrix_t Q_final(4, 4);
        Q.setZero();
        Q(0, 0) = 1;
        Q(1, 1) = 1;
        Q(2, 2) = 1;
        Q(3, 3) = 1;
        Q_final.setZero();
        Q_final(0, 0) = 100;
        Q_final(1, 1) = 100;
        Q_final(2, 2) = 100;
        Q_final(3, 3) = 100;
        ad_matrix_t R(num_control-2, num_control-2);
        R.setZero();
        R(0, 0) = 0.01;
        R(1, 1) = 0.01;
        R(2, 2) = 0.01;
        R(3, 3) = 0.01;
        R(4, 4) = 0.01;
        R(5, 5) = 0.01;
        R(6, 6) = 0.01;
        R(7, 7) = 0.01;

        if (i == N - 1) {
            ad_vector_t tracking_error(4);

            tracking_error << px_i - p[0],
                            py_i - p[1],
                            c_theta - p[2],
                            s_theta - p[3];

            tracking_cost += tracking_error.transpose() * Q_final * tracking_error;
        }

        if (i < N - 1) {
            ad_vector_t control_error(num_control-2);
            control_error << lambda1_i,
                            lambda2_i,
                            lambda3_i,
                            lambda4_i,
                            lambda5_i,
                            lambda6_i,
                            lambda7_i,
                            lambda8_i;
            control_cost += control_error.transpose() * R * control_error;
            ad_vector_t tracking_error(4);
            tracking_error << px_i - p[0],
                            py_i - p[1],
                            c_theta - p[2],
                            s_theta - p[3];
    
            tracking_cost += tracking_error.transpose() * Q * tracking_error;
        }
    }
    y[0] = tracking_cost + control_cost;
};

int main(){
    size_t variableNum = N * (num_state + num_control);
    std::string problemName = "PushT";
    std::string folderName = "model";
    OptimizationProblem pushTProblem(variableNum, problemName);

    auto obj = std::make_shared<ObjectiveFunction>(variableNum, 4, problemName, folderName, "pushTObjective", pushTObjective);
    auto dynamics = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTDynamicConstraints", pushTDynamicConstraints);
    auto contact = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTContactConstraints", pushTContactConstraints);
    auto initial = std::make_shared<ConstraintFunction>(variableNum, 4, problemName, folderName, "pushTInitialConstraints", pushTInitialConstraints);
    auto contactSingleForce = std::make_shared<ConstraintFunction>(variableNum, problemName, folderName, "pushTContactSingleForceConstraints", pushTContactSingleForceConstraints);

    // ---------------------- ! the above four lines are enough for generate the auto-differentiation functions library for this problem and the usage in python ! ---------------------- //

    pushTProblem.addObjective(obj);
    pushTProblem.addEqualityConstraint(dynamics);
    pushTProblem.addEqualityConstraint(initial);
    pushTProblem.addInequalityConstraint(contact);
    pushTProblem.addInequalityConstraint(contactSingleForce);


    // problem parameters
    vector_t xInitialStates(4);
    vector_t xFinalStates(4);
    vector_t xInitialGuess(variableNum);
    vector_t xOptimal(variableNum);
    xFinalStates << 0.01, 0.01, std::cos(0.01), std::sin(0.01);
    // zero initial guess
    xInitialGuess.setZero();
    SolverParameters params;
    SolverInterface solver(pushTProblem, params);
    // Suggestions:
    //
    // 1. Weighted mode may improve convergence, but it may also require more iterations sometimes.
    //    It is worth trying when some groups converge to infeasible solutions.
    //
    // 2. For this problem, an extremely small constraint violation may not be necessary.
    //    A small violation usually only appears in one or two constraints, and the whole
    //    trajectory remains smooth without abrupt jumps. Therefore, it can still work well
    //    when combined with MPC. The user can verify this through visualization by running:
    //    python3 src/examples/pushT/visualize_pushT.py --batch --show-contact
    //    after obtaining the solution. 

    //
    // 3. During an outer loop, the constraints may already be satisfied, but the solver may
    //    continue taking extra iterations to further reduce the objective. At this point,
    //    the objective is often already small enough, and the T block is already close to
    //    the target. If computation speed is important, the user can terminate the solver
    //    early once the constraints are satisfied and the objective is sufficiently low.
    //    This can be confirmed by openning verbose mode.

    // solver.setHyperParameters("WeightedMode", vector_t::Constant(1, 1));
    // solver.setHyperParameters("mu", vector_t::Constant(1, 1));
    solver.setHyperParameters("trailTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("trustRegionTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("constraintTol", vector_t::Constant(1, 1e-3));
    solver.setHyperParameters("verbose", vector_t::Constant(1, 0));

    const size_t numSegments = 50;
    const scalar_t radius_min = 0.25;
    const scalar_t radius_max = 0.5;

    solver.setProblemParameters("pushTObjective", xFinalStates);
    // initialize once, then reuse solver with resetProblem for subsequent segments
    solver.initialize(xInitialGuess);
    size_t lastStepIdx = (N - 1) * (num_state + num_control);
    for (size_t seg = 0; seg < numSegments; ++seg) {
        scalar_t angle = static_cast<scalar_t>(2.0 * M_PI * static_cast<scalar_t>(seg) / static_cast<scalar_t>(numSegments));
        scalar_t radius = radius_min;
        if (numSegments > 1) {
            radius += (radius_max - radius_min) * static_cast<scalar_t>(seg) / static_cast<scalar_t>(numSegments - 1);
        }
        xInitialStates << radius * std::cos(angle),
                          radius * std::sin(angle),
                          std::cos(angle),
                          std::sin(angle);

        solver.setProblemParameters("pushTInitialConstraints", xInitialStates);
        if (seg > 0) {
            solver.resetProblem(xInitialGuess);
        }
        auto segmentStart = std::chrono::high_resolution_clock::now();
        solver.solve();
        auto segmentEnd = std::chrono::high_resolution_clock::now();
        xOptimal = solver.getSolutionSilent();

        const scalar_t finalX = xOptimal[lastStepIdx + 0];
        const scalar_t finalY = xOptimal[lastStepIdx + 1];
        const scalar_t targetX = xFinalStates[0];
        const scalar_t targetY = xFinalStates[1];
        const scalar_t solveTimeMs = static_cast<scalar_t>(
            std::chrono::duration_cast<std::chrono::milliseconds>(segmentEnd - segmentStart).count()
        );
        const scalar_t finalRadius = std::sqrt(finalX * finalX + finalY * finalY);
        const scalar_t targetRadius = std::sqrt(targetX * targetX + targetY * targetY);
        const scalar_t radiusErrorAbs = std::abs(finalRadius - targetRadius);
        const scalar_t posError = std::sqrt((finalX - targetX) * (finalX - targetX) + (finalY - targetY) * (finalY - targetY));

        std::cout << "seg " << (seg + 1) << "/" << numSegments << " final_error | solve_time_ms: " << solveTimeMs
                  << " pos_error: " << posError
                  << " radius_error_abs: " << radiusErrorAbs
                  << std::endl;

        std::ostringstream segFileName;
        segFileName << "pushT_solution_seg_" << std::setw(2) << std::setfill('0') << seg << ".txt";
        saveTrajectoryToTextFile(xOptimal, N, num_state + num_control, segFileName.str());

        if (seg == 0) {
            const std::string defaultTrajectoryFile = "pushT_solution.txt";
            saveTrajectoryToTextFile(xOptimal, N, num_state + num_control, defaultTrajectoryFile);
        }
    }
}
