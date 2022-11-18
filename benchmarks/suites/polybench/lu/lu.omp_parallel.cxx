// BUILD: add_benchmark(ppm=omp_parallel,sources=[__file__,"lu-common.cxx"])

#include "lu-common.h"
#include <rosetta.h>


/*
#pragma minimal dependence distance: 1
for (int c0 = 0; c0 < n - 1; c0 += 1) {
    #pragma simd
    #pragma known-parallel
    for (int c1 = c0 + 1; c1 < n; c1 += 1)
        Stmt_for_cond_cleanup7_i_us(c1, c0);
    #pragma omp parallel for
    for (int c1 = c0 + 1; c1 < n; c1 += 1)
        #pragma simd
        for (int c2 = 0; c2 < n - c1; c2 += 1)
            Stmt_for_body8_i_us(c1, c2, c0);
    #pragma omp parallel for
    for (int c1 = c0 + 2; c1 < n; c1 += 1)
        #pragma simd
        for (int c2 = c0 + 1; c2 < c1; c2 += 1)
            Stmt_for_body43_us_i_us(c1, c2, c0);
}
*/
void kernel_polly(pbsize_t n, multarray<real, 2> A) {
  for (idx_t i = 0; i < n; i++) {
    for (idx_t j = 0; j < i; j++) {
      for (idx_t k = 0; k < j; k++)
      T:
        A[i][j] -= A[i][k] * A[k][j];
    U:
      A[i][j] /= A[j][j];
    }
    for (idx_t j = i; j < n; j++)
      for (idx_t k = 0; k < i; k++)
      V:
        A[i][j] -= A[i][k] * A[k][j];
  }
}



static void
kernel(pbsize_t n, multarray<real, 2> A) {
#pragma omp parallel default(none) firstprivate(n, A)
  {
    for (idx_t k = 0; k < n - 1; k++) {
#pragma omp for schedule(static)
      for (idx_t i = k + 1; i < n; i++) {
      U:
        A[i][k] /= A[k][k];
      }



#pragma omp for collapse(2) /* schedule(static) */
      for (idx_t i = k + 1; i < n; i++)
        for (idx_t j = k + 1; j < n; j++) {
        T:
          A[i][j] -= A[i][k] * A[k][j];
        }
    }
  }
  // U(1,0)
  // V(1,0,0)
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000


  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



  for (auto &&_ : state.manual()) {
    ensure_fullrank(n, A);
    {
      auto &&scope = _.scope();
      kernel(n, A);
    }
  }
}
