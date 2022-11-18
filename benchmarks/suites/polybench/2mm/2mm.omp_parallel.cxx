// BUILD: add_benchmark(ppm=omp_parallel)
#include "rosetta.h"



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
                   real alpha, real beta,
                   multarray<real, 2> tmp,
                   multarray<real, 2> A,
                   multarray<real, 2> B, multarray<real, 2> C, multarray<real, 2> D) {
/* D := alpha*A*B*C + beta*D */
#pragma omp parallel
  {
#pragma omp for collapse(2) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++) {
        tmp[i][j] = 0;
        for (idx_t k = 0; k < nk; ++k)
          tmp[i][j] += alpha * A[i][k] * B[k][j];
      }
#pragma omp for collapse(2) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nl; j++) {
        D[i][j] *= beta;
        for (idx_t k = 0; k < nj; ++k)
          D[i][j] += tmp[i][k] * C[k][j];
      }
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 12; // 1100
  pbsize_t nl = pbsize;               // 1200

  real alpha = 1.5;
  real beta = 1.2;
  auto tmp = state.allocate_array<real>({ni, nj}, /*fakedata*/ false, /*verify*/ false, "tmp");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto C = state.allocate_array<real>({nj, nl}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({ni, nl}, /*fakedata*/ true, /*verify*/ true, "D");

  for (auto &&_ : state)
    kernel(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
}
