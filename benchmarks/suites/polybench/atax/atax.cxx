// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"

static void kernel(pbsize_t m, pbsize_t n, multarray<real, 2> A, real x[], real y[], real tmp[]) {
#pragma scop
  for (idx_t i = 0; i < n; i++)
    y[i] = 0;
  for (idx_t i = 0; i < m; i++) {
    tmp[i] = 0;
    for (idx_t j = 0; j < n; j++)
      tmp[i] += A[i][j] * x[j];
    for (idx_t j = 0; j < n; j++)
      y[j] += A[i][j] * tmp[i];
  }
#pragma endscop
}

 

void run(State &state, pbsize_t pbsize) {
    // n is 5%-20% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 10;

  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");
  auto tmp = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ false, "tmp");

  for (auto &&_ : state)
    kernel(m, n, A, x, y, tmp);
}
