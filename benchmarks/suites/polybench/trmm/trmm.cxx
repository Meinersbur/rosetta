// BUILD: add_benchmark(ppm=serial)

#include <rosetta.h>

/*
      #pragma omp parallel for
      for (int c0 = 0; c0 < n; c0 += 1)
        #pragma minimal dependence distance: 1
        for (int c1 = 0; c1 < m - 1; c1 += 1)
          #pragma simd
          for (int c2 = 0; c2 <= c1; c2 += 1)
            Stmt_for_body8(c2, c0, c1 - c2);
      #pragma omp parallel for
      for (int c0 = 0; c0 < m; c0 += 1)
        #pragma simd
        for (int c1 = 0; c1 < n; c1 += 1)
          Stmt_for_end(c0, c1);
*/
#if 0
extern "C" void kernel_polly(pbsize_t n, pbsize_t m,
    real alpha,
    real B[][n], real A[][m] ) {
#pragma scop
    for (idx_t i = 0; i < m; i++)
        for (idx_t j = 0; j < n; j++) {
            for (idx_t k = i + 1; k < m; k++)
                B[i][j] += A[k][i] * B[k][j];
            B[i][j] *= alpha ;
        }
#pragma endscop
}
#endif



static void kernel(pbsize_t n, pbsize_t m,
                   real alpha,
                   multarray<real, 2> B, multarray<real, 2> A) {
#pragma scop
  for (idx_t i = 0; i < m; i++)
    for (idx_t j = 0; j < n; j++) {
      for (idx_t k = i + 1; k < m; k++)
        B[i][j] += A[k][i] * B[k][j];
      B[i][j] *= alpha;
    }
#pragma endscop
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "B");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");

  for (auto &&_ : state)
    kernel(n, m, alpha, B, A);
}
