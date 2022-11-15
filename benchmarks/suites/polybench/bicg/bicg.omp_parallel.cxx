// BUILD: add_benchmark(ppm=omp_parallel)
#include "rosetta.h"


static void kernel(pbsize_t m, pbsize_t n, multarray<real, 2> A, real s[], real q[], real p[], real r[]) {
  #pragma omp parallel
  {
       #pragma omp for schedule(static) nowait
  for (idx_t i = 0; i < n; i++) {
    q[i] = 0;
    for (idx_t j = 0; j < m; j++)
      q[i] += A[i][j] * p[j];
  }
      #pragma omp for  schedule(static) nowait
            for (idx_t j = 0; j < m; j++) {
                  s[j] = 0;
      for (idx_t i = 0; i < n; i++)
        s[j] += r[i] * A[i][j];
            }
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t m = pbsize - 19 * pbsize / 21; // 1900
  pbsize_t n = pbsize;                    // 2100

  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto s = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "s");
  auto q = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "q");
  auto p = state.allocate_array<real>({m}, /*fakedata*/ true, /*verify*/ false, "p");
  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");



  for (auto &&_ : state)
    kernel(m, n, A, s, q, p, r);
}
