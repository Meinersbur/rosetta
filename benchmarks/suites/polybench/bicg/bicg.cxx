// BUILD: add_benchmark(ppm=serial)


#include "rosetta.h"


static void kernel(int m, int n, multarray<real, 2> A, real s[], real q[], real p[], real r[]) {
#pragma scop
  for (int i = 0; i < m; i++)
    s[i] = 0;
  for (int i = 0; i < n; i++) {
    q[i] = 0;
    for (int j = 0; j < m; j++) {
      s[j] = s[j] + r[i] * A[i][j];
      q[i] = q[i] + A[i][j] * p[j];
    }
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t m = pbsize - 19 * pbsize / 21; // 1900
  size_t n = pbsize;                    // 2100



  auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);
  auto s = state.allocate_array<double>({m}, /*fakedata*/ false, /*verify*/ true);
  auto q = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true);
  auto p = state.allocate_array<double>({m}, /*fakedata*/ true, /*verify*/ false);
  auto r = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(m, n, A, s, q, p, r);
}
