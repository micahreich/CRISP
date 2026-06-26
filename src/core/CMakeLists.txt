# add the solver as a library, the solver only has one cpp file and others are header files only.
add_library(CRISP
  src/cppad_core/CppAdInterface.cpp
)

target_include_directories(CRISP PUBLIC
  include
  ${MATLAB_ROOT}/extern/include                    
  ${EIGEN3_INCLUDE_DIR}     
  ${Boost_INCLUDE_DIRS}
  ${CPPAD_INCLUDE_DIRS}
  ${CPPADCG_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}      
)

target_link_directories(CRISP PUBLIC
  ${CPPAD_LIBRARY_DIRS}
  ${CPPADCG_LIBRARY_DIRS}
)

target_link_libraries(CRISP
  ${Boost_LIBRARIES}
  ${EIGEN3_LIBRARIES}
  ${CPPAD_LIBRARIES}
  yaml-cpp
  piqp::piqp # the piqp library (header file library)
  -ldl
)

# # level 1 test, test the cppad interface: PASSED
# add_executable(test_core_cppad tests/test_cppad_interface.cpp)


# target_link_libraries(test_core_cppad
#   ContactSolver
#   -ldl
# )

# # level 2 test, test the core functions for constructing the optimization problem: PASSED
# add_executable(test_core_optimizationProblem tests/test_optimizationProblem.cpp)

# target_link_libraries(test_core_optimizationProblem
#   ContactSolver
#   -ldl
# )

# # level 3 test, test the core functions for solving the optimization problem: PASSED
# add_executable(test_core_solver tests/test_solver.cpp)
# target_link_libraries(test_core_solver
#   ContactSolver
#   -ldl
# )

# test pybind
# pybind11_add_module(pybind_test python/testPybind.cpp)
# install(TARGETS pybind_test DESTINATION ${CMAKE_SOURCE_DIR}/python)

pybind11_add_module(pyCRISP python/pybindSolver.cpp)
# Linking the static library
target_link_libraries(pyCRISP PRIVATE CRISP)
# Ensure position-independent code for the shared module
set_target_properties(CRISP pyCRISP PROPERTIES POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED JlCxx_DIR)
  find_program(JULIA_EXECUTABLE julia)
  if(JULIA_EXECUTABLE)
    execute_process(
      COMMAND ${JULIA_EXECUTABLE} --startup-file=no -e "using CxxWrap; print(CxxWrap.prefix_path())"
      OUTPUT_VARIABLE _jlcxx_prefix
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    if(_jlcxx_prefix)
      set(JlCxx_DIR "${_jlcxx_prefix}/lib/cmake/JlCxx"
          CACHE PATH "Path to JlCxxConfig.cmake auto-detected from Julia")
      message(STATUS "Auto-detected JlCxx_DIR: ${JlCxx_DIR}")
    else()
      message(WARNING "Could not detect JlCxx from Julia. Install CxxWrap.jl or set JlCxx_DIR to build Julia bindings.")
    endif()
  endif()
endif()

find_package(JlCxx QUIET)
if(JlCxx_FOUND)
  add_library(crisp_julia SHARED
    julia/crisp_julia_bindings.cpp
  )
  target_link_libraries(crisp_julia PRIVATE CRISP JlCxx::cxxwrap_julia)
  set_target_properties(crisp_julia PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
  )
else()
  message(WARNING "JlCxx not found; skipping CRISP Julia CxxWrap bindings.")
endif()
