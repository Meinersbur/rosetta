// BUILD: add_benchmark(ppm=serial)

#include <rosetta.h>



static void kernel(pbsize_t n, pbsize_t m,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A) {
#pragma scop
  for (idx_t i = 0; i < n; i++) {
    for (idx_t j = 0; j <= i; j++)
      C[i][j] *= beta;
    for (idx_t k = 0; k < m; k++)
      for (idx_t j = 0; j <= i; j++)
        C[i][j] += alpha * A[i][k] * A[j][k];
  }
#pragma endscop
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");



  for (auto &&_ : state)
    kernel(n, m, alpha, beta, C, A);
}
