# CRISP Julia interface

Minimal [CxxWrap](https://github.com/JuliaInterop/CxxWrap.jl) bindings that let
you call CRISP's solver from Julia.

## Build

CRISP's Julia bindings build automatically as part of the normal CRISP build
whenever CxxWrap.jl is installed — `JlCxx_DIR` is auto-detected from your Julia
installation. From the repository root:

```bash
cmake -S . -B build
cmake --build build --target crisp_julia -j4
```

(install CxxWrap first with `julia -e 'using Pkg; Pkg.add("CxxWrap")'`). The
compiled library is written to `build/lib/libcrisp_julia.*`, which is where the
Julia package looks for it by default. If you build elsewhere, point it there
with a `LocalPreferences.toml` entry `libcrisp_julia_path`.

Then instantiate the Julia project:

```bash
julia --project=julia -e 'using Pkg; Pkg.instantiate()'
```

## Usage

The solver takes QPCC data in "Marble" form as a `NamedTuple`:

```
minimize    1/2 x'Q x + q'x + c0
subject to  J_eq x + b_eq == 0
            J_ineq x + b_ineq >= 0
            0 <= L x + l   complements   R x + r >= 0
```

```julia
using Pkg; Pkg.activate("julia")
using CRISP

# minimize 1/2 x'(2I)x  s.t.  x1 == 1,  x2 >= 1,  0 <= x3+1 comp x4-1 >= 0
data = (
    Q = [2.0 0 0 0; 0 2 0 0; 0 0 2 0; 0 0 0 2],
    q = zeros(4),
    c0 = 0.0,
    J_eq   = [1.0 0 0 0], b_eq   = [-1.0],   # x1 - 1 == 0
    J_ineq = [0.0 1 0 0], b_ineq = [-1.0],   # x2 - 1 >= 0
    L = [0.0 0 1 0], l = [1.0],              # 0 <= x3 + 1
    R = [0.0 0 0 1], r = [-1.0],             #      complements  x4 - 1 >= 0
)

result = CRISP.solve_qpcc_with_crisp(data;
    trust_region_tol = 1e-4,
    constraint_tol = 1e-6,
    verbose = 0,
)

result.converged   # true
result.x           # [1.0, 1.0, 0.0, 1.0]
result.iterations
result.solve_time_seconds
```

All eleven fields `Q`, `q`, `c0`, `J_eq`, `b_eq`, `J_ineq`, `b_ineq`, `L`, `l`,
`R`, `r` are required; pass zero-row blocks (e.g. `zeros(0, n)`) for constraint
types you don't have. An optional primal warm start is passed with the `x0`
keyword.

`solve_qpcc_with_crisp` returns a `NamedTuple` with fields:

| field                | meaning                                                        |
| -------------------- | -------------------------------------------------------------- |
| `converged`          | `true` iff CRISP's solver reported convergence                 |
| `x`                  | primal solution (length `n`)                                   |
| `setup_time_seconds` | problem construction, incl. CppAD code generation (see below)  |
| `solve_time_seconds` | wall-clock time in the solver loop (excludes CppAD generation) |
| `qp_time_seconds`    | time spent in the QP subsolver                                 |
| `iterations`         | number of solver iterations                                    |

### CppAD code generation

CRISP differentiates the problem with CppAD, generating and compiling a
derivative library. Two `solve_qpcc_with_crisp` keyword arguments control this:

- `regenerate_library` (default `true`) — regenerate the library on each call.
  Set to `false` to reuse a previously generated one, which makes
  `setup_time_seconds` drop from seconds to near-zero.
- `problem_name` / `folder_name` — the name and location of the generated
  library (defaults to the repository's `model/` folder).

Generation cost lands in `setup_time_seconds`; `solve_time_seconds` measures only
the solver loop and never includes it.

### Options

Other solver hyperparameters are snake-case keyword arguments (see
`CRISP.OPTION_NAMES`), e.g. `max_iterations`, `trust_region_init_radius`,
`trust_region_max_radius`, `mu`, `mu_max`, `eta_low`, `eta_high`, `trail_tol`,
`trust_region_tol`, `constraint_tol`, `weighted_mode`, `weighted_tol_factor`,
`second_order_correction`, and `verbose`. Any option not passed keeps CRISP's
default.

See `examples/simple_qpcc.jl` for a runnable example.
