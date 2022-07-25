#include "rosetta.h"


static void kernel(int m, int n,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A,
                   multarray<real, 2> B) {
#pragma scop
  for (int i = 0; i < m; i++)
    for (int j = 0; j < n; j++) {
      real temp2 = 0;
      for (int k = 0; k < i; k++) {
        C[k][j] += alpha * B[i][j] * A[i][k];
        temp2 += B[k][j] * A[i][k];
      }
      C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * temp2;
    }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t n = pbsize;
  size_t m = pbsize - pbsize / 8;


  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({m, n}, /*fakedata*/ false, /*verify*/ true);
  auto A = state.allocate_array<double>({m, m}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({m, n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(m, n, alpha, beta, C, A, B);
}
