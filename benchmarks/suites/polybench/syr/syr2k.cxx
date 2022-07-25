#include "rosetta.h"



static void kernel(int n, int m,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A,
                   multarray<real, 2> B) {
#pragma scop
  for (int i = 0; i < n; i++) {
    for (int j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (int k = 0; k < m; k++)
      for (int j = 0; j <= i; j++) {
        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];
      }
  }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t n = pbsize;
  size_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true);
  auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(n, m, alpha, beta, C, A, B);
}
