// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t n, real alpha, real beta, multarray<real, 2> A, real u1[], real v1[], real u2[], real v2[], real w[], real x[], real y[], real z[]) {
  real *pA = &A[0][0];

#pragma omp target data map(to                                                                     \
                            : y [0:n], z [0:n], u1 [0:n], v1 [0:n], u2 [0:n], v2 [0:n]) map(tofrom \
                                                                                            : pA [0:n * n], w [0:n], x [0:n])
  {

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < n; j++)
        pA[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < n; j++)
        x[i] += beta * pA[j * n + i] * y[j];

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < n; i++)
      x[i] += z[i];

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j < n; j++)
        w[i] += alpha * pA[i * n + j] * x[j];
  }
}


void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto y = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "y");
  auto z = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "z");
  auto u1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u1");
  auto v1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v1");
  auto u2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u2");
  auto v2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v2");
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto w = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "w");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "x");



  for (auto &&_ : state)
    kernel(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
}
