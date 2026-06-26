# Solving Marble Matrix Data With CRISP

This example loads the binary problem format written by Marble's Julia
`write_problem` helper:

```julia
data = (; Q, q, c0, J_eq, b_eq, J_ineq, b_ineq, L, l, R, r)
Marble.write_problem("problem.marble", data)
```

The loaded problem is interpreted as

```text
minimize    1/2 x'Qx + q'x + c0
subject to  J_eq x + b_eq == 0
            J_ineq x + b_ineq >= 0
            0 <= Lx + l  complements  Rx + r >= 0
```

CRISP receives the complementarity block as ordinary nonlinear inequalities:
`Lx + l >= 0`, `Rx + r >= 0`, and `-(Lx + l).*(Rx + r) >= 0`.

To generate the tiny bundled example data:

```sh
julia src/examples/marble/write_simple_problem.jl
```

Then build and solve:

```sh
cmake -S src -B build
cmake --build build --target marble_data_example
./build/examples/marble_data_example src/examples/marble/simple_qpcc.marble
```

You can also pass an initial guess text file and a solution output path:

```sh
./build/examples/marble_data_example problem.marble x0.txt solution.txt
```
