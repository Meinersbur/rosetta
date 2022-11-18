// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"


static void kernel(pbsize_t n,
                   real alpha, real beta,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   real tmp[],
                   real x[],
                   real y[]) {
#pragma scop
  for (idx_t i = 0; i < n; i++) {
    tmp[i] = 0;
    y[i] = 0;
    for (idx_t j = 0; j < n; j++) {
      tmp[i] += A[i][j] * x[j];
      y[i] += B[i][j] * x[j];
    }
    y[i] = alpha * tmp[i] + beta * y[i];
  }
#pragma endscop
}



void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false);
  auto tmp = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false);
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false);
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true);



  for (auto &&_ : state)
    kernel(n, alpha, beta, A, B, tmp, x, y);
}
