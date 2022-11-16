// BUILD: add_benchmark(ppm=serial,sources=[__file__,"lu-common.cxx"])

#include <rosetta.h>
#include "lu-common.h"



/*
#pragma minimal dependence distance: 1
for (int c0 = 0; c0 < p_0 - 1; c0 += 1) {
#pragma simd
#pragma known-parallel
for (int c1 = c0 + 1; c1 < p_0; c1 += 1)
Stmt6(c1, c0);
#pragma omp parallel for
for (int c1 = c0 + 2; c1 < p_0; c1 += 1)
#pragma simd
for (int c2 = c0 + 1; c2 < c1; c2 += 1)
Stmt5(c1, c2, c0);
#pragma omp parallel for
for (int c1 = c0 + 1; c1 < p_0; c1 += 1)
#pragma simd
for (int c2 = 0; c2 < p_0 - c1; c2 += 1)
Stmt8(c1, c2, c0);
}
*/
#if 0
__attribute__((noinline))
 extern "C" void kernel_polly(pbsize_t  n,  real A[][n]) {

    for (idx_t i = 0; i < n; i++) {
        for (idx_t j = 0; j < i; j++) {
            for (idx_t k = 0; k < j; k++)
                A[i][j] -= A[i][k] * A[k][j];
            A[i][j] /= A[j][j];
        }
        for (idx_t j = i; j < n; j++) 
            for (idx_t k = 0; k < i; k++)
                A[i][j] -= A[i][k] * A[k][j];

    }
    
}
#endif 

__attribute__((noinline))
 void kernel(pbsize_t  n,  multarray<real, 2> A) {
#pragma scop 
  for (idx_t i = 0; i < n; i++) {
    for (idx_t j = 0; j < i; j++) {
        for (idx_t k = 0; k < j; k++) 
             A[i][j] -= A[i][k] * A[k][j];
U:      A[i][j] /= A[j][j];
    }
    for (idx_t j = i; j < n; j++) 
        for (idx_t k = 0; k < i; k++) 
           A[i][j] -= A[i][k] * A[k][j];
  }
#pragma endscop


  // U(1,0)
  // V(1,1,0)
}




void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



    for (auto&& _ : state.manual()) {
        ensure_fullrank(n, A);
        {
            auto &&scope = _.scope();
            kernel(n, A);
        }
    }
}
