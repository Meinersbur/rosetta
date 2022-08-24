// BUILD: add_benchmark(ppm=omp_target)
#include "rosetta.h"


static void kernel(int m, int n, multarray<real, 2> A, real *x, real *y, real *tmp) {
  #pragma omp target map(alloc:tmp) map(in:A,x) map(out:y) 
  {
#pragma omp teams distribute parallel for
  for (int i = 0; i < n; i++)
    y[i] = 0;
#pragma omp teams distribute parallel for
  for (int i = 0; i < m; i++) {
    tmp[i] = 0;
#pragma omp teams distribute parallel for collapse(2)
    for (int i = 0; i < m; i++) 
    for (int j = 0; j < n; j++)
      tmp[i] +=  A[i][j] * x[j];
 
#pragma omp teams distribute parallel for collapse(2)
   for (int j = 0; j < n; j++)
        for (int i = 0; i < m; i++)
      y[j] +=  A[i][j] * tmp[i];
}


void run(State &state, int pbsize) {
  // n is 5%-20% larger than m
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 10;

  auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);
  auto x = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false);
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true);
  auto tmp = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ false);

  for (auto &&_ : state)
    kernel(n, m, A, x, y, tmp);
}
