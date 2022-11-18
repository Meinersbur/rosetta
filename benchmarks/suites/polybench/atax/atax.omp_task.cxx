// BUILD: add_benchmark(ppm=omp_task)
#include "rosetta.h"


static void kernel(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
#pragma omp parallel
  {
#pragma omp single
    {
#pragma omp taskloop
      for (int i = 0; i < n; i++)
        y[i] = 0;
#pragma omp taskloop
      for (int i = 0; i < m; i++) {
        tmp[i] = 0;
        for (int j = 0; j < n; j++)
          tmp[i] += A[i][j] * x[j];
      }
#pragma omp taskwait
#pragma omp taskloop
      for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
          y[j] += A[i][j] * tmp[i];
    }
  }
}


void run(State &state, int pbsize) {
  // n is 5%-20% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 10;

  auto A = state.allocate_array<double>({m, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true, "y");
  auto tmp = state.allocate_array<double>({m}, /*fakedata*/ false, /*verify*/ false, "tmp");

  for (auto &&_ : state)
    kernel(m, n, A, x, y, tmp);
}
