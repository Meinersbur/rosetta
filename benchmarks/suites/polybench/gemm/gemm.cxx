#include "rosetta.h"


static void kernel(int ni, int nj, int nk,
                   real alpha,
                   real beta,
                   multarray<real, 2> C, multarray<real, 2> A, multarray<real, 2> B) {
#pragma scop
  for (int i = 0; i < ni; i++) {
    for (int j = 0; j < nj; j++)
      C[i][j] *= beta;
    for (int k = 0; k < nk; k++) {
      for (int j = 0; j < nj; j++)
        C[i][j] += alpha * A[i][k] * B[k][j];
    }
  }
#pragma endscop
}


void run(State &state, int pbsize) {
  size_t ni = pbsize - pbsize / 4;
  size_t nj = pbsize - pbsize / 8;
  size_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({ni, nj}, /*fakedata*/ true, /*verify*/ true);
  auto A = state.allocate_array<double>({ni, nk}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({nk, nj}, /*fakedata*/ true, /*verify*/ false);

  for (auto &&_ : state)
    kernel(ni, nj, nk, alpha, beta, C, A, B);
}
