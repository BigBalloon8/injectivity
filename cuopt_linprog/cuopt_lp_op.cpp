// cuopt_lp_op.cpp
//
// Custom PyTorch C++ operator wrapping NVIDIA cuOpt's LP solver (C API).
//
// Registers torch.ops.cuopt_lp.solve_batch, which solves a batch of LPs of
// the ranged form
//
//     minimize    c^T x
//     subject to  con_lb <= A x <= con_ub      (A in CSR format)
//                 var_lb <=   x <= var_ub
//
// by looping over cuOptSolve calls (the cuOpt C API is per-problem; the GPU
// work happens inside each solve). The loop runs with the GIL released.
// Batch fusion via block-diagonal stacking is done by the Python wrapper,
// which then calls this op with batch size 1.
//
// Build: see setup.py / cuopt_linprog.load_extension(). Requires libcuopt
// (https://github.com/NVIDIA/cuopt, Apache-2.0) and its cuopt_c.h header.

#include <torch/extension.h>

#include <cuopt/linear_programming/cuopt_c.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace {

// scipy.optimize.linprog status codes
constexpr int64_t kOptimal = 0;
constexpr int64_t kIterationLimit = 1;
constexpr int64_t kInfeasible = 2;
constexpr int64_t kUnbounded = 3;
constexpr int64_t kNumericalTrouble = 4;

int64_t map_termination_status(cuopt_int_t t) {
  switch (t) {
    case CUOPT_TERMINATION_STATUS_OPTIMAL:
      return kOptimal;
    case CUOPT_TERMINATION_STATUS_ITERATION_LIMIT:
    case CUOPT_TERMINATION_STATUS_TIME_LIMIT:
    case CUOPT_TERMINATION_STATUS_CONCURRENT_LIMIT:
    case CUOPT_TERMINATION_STATUS_PRIMAL_FEASIBLE:
    case CUOPT_TERMINATION_STATUS_FEASIBLE_FOUND:
      return kIterationLimit;  // stopped early; solution may be usable
    case CUOPT_TERMINATION_STATUS_INFEASIBLE:
      return kInfeasible;
    case CUOPT_TERMINATION_STATUS_UNBOUNDED:
      return kUnbounded;
    default:  // NUMERICAL_ERROR, UNBOUNDED_OR_INFEASIBLE, NO_TERMINATION, ...
      return kNumericalTrouble;
  }
}

// RAII wrappers so early exits can't leak cuOpt objects.
struct Problem {
  cuOptOptimizationProblem h = nullptr;
  ~Problem() {
    if (h) cuOptDestroyProblem(&h);
  }
};
struct SolverSettings {
  cuOptSolverSettings h = nullptr;
  ~SolverSettings() {
    if (h) cuOptDestroySolverSettings(&h);
  }
};
struct Solution {
  cuOptSolution h = nullptr;
  ~Solution() {
    if (h) cuOptDestroySolution(&h);
  }
};

// Replace +-inf with cuOpt's infinity sentinel.
void copy_with_infinity(const double* src, std::vector<cuopt_float_t>& dst,
                        int64_t count) {
  dst.resize(count);
  for (int64_t i = 0; i < count; ++i) {
    double v = src[i];
    if (std::isinf(v)) {
      dst[i] = v > 0 ? CUOPT_INFINITY : -CUOPT_INFINITY;
    } else {
      dst[i] = static_cast<cuopt_float_t>(v);
    }
  }
}

void check_cpu_contig(const at::Tensor& t, const char* name,
                      at::ScalarType dtype) {
  TORCH_CHECK(t.device().is_cpu(), name, " must be a CPU tensor");
  TORCH_CHECK(t.is_contiguous(), name, " must be contiguous");
  TORCH_CHECK(t.scalar_type() == dtype, name, " must have dtype ",
              dtype, ", got ", t.scalar_type());
}

}  // namespace

// c        : (B, n)  float64
// row_off  : (m + 1,) int32   CSR row offsets (shared across the batch)
// col_idx  : (nnz,)   int32   CSR column indices (shared across the batch)
// values   : (B, nnz) float64 CSR values (expand a shared A to B rows;
//                              expand() makes this free — no copy)
// con_lb/ub: (B, m)   float64 constraint bounds (+-inf allowed)
// var_lb/ub: (B, n)   float64 variable bounds (+-inf allowed)
// tol      : relative primal/dual/gap tolerance passed to cuOpt
// time_limit : per-problem time limit in seconds (<= 0 -> leave default)
// method   : 0 concurrent, 1 PDLP, 2 dual simplex, 3 barrier
// per_constraint_residual : use per-constraint termination criteria
//                           (recommended for block-diagonal stacked solves)
//
// Returns {x (B, n), fun (B,), status (B,), duals (B, m),
//          reduced_costs (B, n), solve_time (B,)}
std::vector<at::Tensor> solve_batch(
    const at::Tensor& c, const at::Tensor& row_off, const at::Tensor& col_idx,
    const at::Tensor& values, const at::Tensor& con_lb,
    const at::Tensor& con_ub, const at::Tensor& var_lb,
    const at::Tensor& var_ub, double tol, double time_limit, int64_t method,
    bool per_constraint_residual) {
  TORCH_CHECK(cuOptGetFloatSize() == sizeof(double),
              "this extension assumes a cuOpt build with 64-bit floats");
  TORCH_CHECK(cuOptGetIntSize() == sizeof(int32_t),
              "this extension assumes a cuOpt build with 32-bit ints");

  check_cpu_contig(c, "c", at::kDouble);
  check_cpu_contig(row_off, "row_offsets", at::kInt);
  check_cpu_contig(col_idx, "col_indices", at::kInt);
  check_cpu_contig(con_lb, "con_lb", at::kDouble);
  check_cpu_contig(con_ub, "con_ub", at::kDouble);
  check_cpu_contig(var_lb, "var_lb", at::kDouble);
  check_cpu_contig(var_ub, "var_ub", at::kDouble);
  TORCH_CHECK(values.device().is_cpu() && values.scalar_type() == at::kDouble,
              "values must be a CPU float64 tensor");

  const int64_t B = c.size(0);
  const int64_t n = c.size(1);
  const int64_t m = con_lb.size(1);
  const int64_t nnz = col_idx.size(0);
  TORCH_CHECK(row_off.size(0) == m + 1, "row_offsets must have m + 1 entries");
  TORCH_CHECK(values.dim() == 2 && values.size(0) == B &&
                  values.size(1) == nnz,
              "values must have shape (B, nnz)");
  TORCH_CHECK(con_ub.sizes() == con_lb.sizes() && con_lb.size(0) == B,
              "constraint bounds must have shape (B, m)");
  TORCH_CHECK(var_lb.size(0) == B && var_lb.size(1) == n &&
                  var_ub.sizes() == var_lb.sizes(),
              "variable bounds must have shape (B, n)");
  TORCH_CHECK(n <= std::numeric_limits<int32_t>::max() &&
                  m <= std::numeric_limits<int32_t>::max() &&
                  nnz <= std::numeric_limits<int32_t>::max(),
              "problem dimensions exceed cuOpt's 32-bit index range");

  // values rows may be a zero-stride expand of one shared A — handle both.
  const bool shared_values = (values.stride(0) == 0);
  const at::Tensor values_c =
      shared_values ? values.select(0, 0).contiguous() : values.contiguous();

  auto x_out = at::full({B, n}, std::numeric_limits<double>::quiet_NaN(),
                        c.options());
  auto fun_out = at::full({B}, std::numeric_limits<double>::quiet_NaN(),
                          c.options());
  auto status_out = at::full({B}, kNumericalTrouble,
                             c.options().dtype(at::kLong));
  auto duals_out = at::full({B, m}, std::numeric_limits<double>::quiet_NaN(),
                            c.options());
  auto rc_out = at::full({B, n}, std::numeric_limits<double>::quiet_NaN(),
                         c.options());
  auto time_out = at::zeros({B}, c.options());

  const double* c_ptr = c.const_data_ptr<double>();
  const int32_t* ro_ptr = row_off.const_data_ptr<int32_t>();
  const int32_t* ci_ptr = col_idx.const_data_ptr<int32_t>();
  const double* val_ptr = values_c.const_data_ptr<double>();
  const double* clb_ptr = con_lb.const_data_ptr<double>();
  const double* cub_ptr = con_ub.const_data_ptr<double>();
  const double* vlb_ptr = var_lb.const_data_ptr<double>();
  const double* vub_ptr = var_ub.const_data_ptr<double>();

  double* x_ptr = x_out.data_ptr<double>();
  double* fun_ptr = fun_out.data_ptr<double>();
  int64_t* st_ptr = status_out.data_ptr<int64_t>();
  double* du_ptr = duals_out.data_ptr<double>();
  double* rc_ptr = rc_out.data_ptr<double>();
  double* tm_ptr = time_out.data_ptr<double>();

  const std::vector<char> var_types(static_cast<size_t>(n), CUOPT_CONTINUOUS);

  const cuopt_int_t method_const =
      method == 1   ? CUOPT_METHOD_PDLP
      : method == 2 ? CUOPT_METHOD_DUAL_SIMPLEX
      : method == 3 ? CUOPT_METHOD_BARRIER
                    : CUOPT_METHOD_CONCURRENT;

  std::vector<cuopt_float_t> clb, cub, vlb, vub;
  std::vector<cuopt_float_t> primal(static_cast<size_t>(n));
  std::vector<cuopt_float_t> dual(static_cast<size_t>(m));
  std::vector<cuopt_float_t> rcost(static_cast<size_t>(n));

  for (int64_t b = 0; b < B; ++b) {
    copy_with_infinity(clb_ptr + b * m, clb, m);
    copy_with_infinity(cub_ptr + b * m, cub, m);
    copy_with_infinity(vlb_ptr + b * n, vlb, n);
    copy_with_infinity(vub_ptr + b * n, vub, n);
    const double* vals_b = shared_values ? val_ptr : val_ptr + b * nnz;

    Problem prob;
    cuopt_int_t st = cuOptCreateRangedProblem(
        static_cast<cuopt_int_t>(m), static_cast<cuopt_int_t>(n),
        CUOPT_MINIMIZE, /*objective_offset=*/0.0, c_ptr + b * n, ro_ptr,
        ci_ptr, vals_b, clb.data(), cub.data(), vlb.data(), vub.data(),
        var_types.data(), &prob.h);
    if (st != CUOPT_SUCCESS) continue;  // status stays 4 (numerical trouble)

    SolverSettings settings;
    if (cuOptCreateSolverSettings(&settings.h) != CUOPT_SUCCESS) continue;
    cuOptSetIntegerParameter(settings.h, CUOPT_LOG_TO_CONSOLE, 0);
    cuOptSetIntegerParameter(settings.h, CUOPT_METHOD, method_const);
    if (tol > 0) {
      cuOptSetFloatParameter(settings.h, CUOPT_RELATIVE_PRIMAL_TOLERANCE, tol);
      cuOptSetFloatParameter(settings.h, CUOPT_RELATIVE_DUAL_TOLERANCE, tol);
      cuOptSetFloatParameter(settings.h, CUOPT_RELATIVE_GAP_TOLERANCE, tol);
    }
    if (time_limit > 0) {
      cuOptSetFloatParameter(settings.h, CUOPT_TIME_LIMIT, time_limit);
    }
    if (per_constraint_residual) {
      cuOptSetIntegerParameter(settings.h, CUOPT_PER_CONSTRAINT_RESIDUAL, 1);
    }

    Solution sol;
    if (cuOptSolve(prob.h, settings.h, &sol.h) != CUOPT_SUCCESS) continue;

    cuopt_int_t term = CUOPT_TERMINATION_STATUS_NO_TERMINATION;
    if (cuOptGetTerminationStatus(sol.h, &term) != CUOPT_SUCCESS) continue;
    st_ptr[b] = map_termination_status(term);

    cuopt_float_t t = 0;
    if (cuOptGetSolveTime(sol.h, &t) == CUOPT_SUCCESS) tm_ptr[b] = t;

    // A primal iterate is meaningful for optimal / limit-terminated solves.
    if (st_ptr[b] == kOptimal || st_ptr[b] == kIterationLimit) {
      cuopt_float_t obj = 0;
      if (cuOptGetObjectiveValue(sol.h, &obj) == CUOPT_SUCCESS) {
        fun_ptr[b] = obj;
      }
      if (cuOptGetPrimalSolution(sol.h, primal.data()) == CUOPT_SUCCESS) {
        for (int64_t j = 0; j < n; ++j) x_ptr[b * n + j] = primal[j];
      }
      if (m > 0 && cuOptGetDualSolution(sol.h, dual.data()) == CUOPT_SUCCESS) {
        for (int64_t i = 0; i < m; ++i) du_ptr[b * m + i] = dual[i];
      }
      if (cuOptGetReducedCosts(sol.h, rcost.data()) == CUOPT_SUCCESS) {
        for (int64_t j = 0; j < n; ++j) rc_ptr[b * n + j] = rcost[j];
      }
    }
  }

  return {x_out, fun_out, status_out, duals_out, rc_out, time_out};
}

TORCH_LIBRARY(cuopt_lp, m) {
  m.def(
      "solve_batch(Tensor c, Tensor row_offsets, Tensor col_indices, "
      "Tensor values, Tensor con_lb, Tensor con_ub, Tensor var_lb, "
      "Tensor var_ub, float tol, float time_limit, int method, "
      "bool per_constraint_residual) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(cuopt_lp, CPU, m) { m.impl("solve_batch", &solve_batch); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "PyTorch custom op wrapping NVIDIA cuOpt's LP solver";
}
