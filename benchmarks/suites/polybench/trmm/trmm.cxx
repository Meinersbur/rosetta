#include "rosetta.h"



static void kernel(int n, int m,
                   real alpha,
                   multarray<real, 2> B, multarray<real, 2> A) {
#pragma scop
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      for (int k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] = alpha * B[i][j];
    }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t n = pbsize;
  size_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<double>({m, n}, /*fakedata*/ true, /*verify*/ true);
  auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);



  for (auto &&_ : state)
    kernel(n, m, alpha, B, A);
}
