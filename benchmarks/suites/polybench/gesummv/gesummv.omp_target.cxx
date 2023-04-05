// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>


static void kernel(pbsize_t n,
                   real alpha, real beta,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   real tmp[],
                   real x[],
                   real y[]) {
    real *pA = &A[0][0];
    real *pB = &B[0][0];

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) map(to:pA[0:n*n],pB[0:n*n],x[0:n]) map(alloc:tmp[0:n]) map(from:y[0:n])
  for (idx_t i = 0; i < n; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (idx_t j = 0; j < n; j++) {
      tmp[i] += A[i*n+j] * x[j];
      y[i] += B[i*n+j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
}



void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "B");
  auto tmp = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false, "tmp");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true, "y");



  for (auto &&_ : state)
    kernel(n, alpha, beta, A, B, tmp, x, y);
}
