// BUILD: add_benchmark(ppm=serial)

#include <rosetta.h>


/*
    {
      #pragma simd
      #pragma known-parallel
      for (int c0 = 0; c0 < n; c0 += 1)
        Stmt_for_body(c0);
      #pragma minimal dependence distance: 1
      for (int c0 = 0; c0 < n; c0 += 1) {
        Stmt_for_end(c0);
        #pragma simd
        #pragma known-parallel
        for (int c1 = c0 + 1; c1 < n; c1 += 1)
          Stmt_for_body6(c1, c0);
      }
    }
*/

#if 0
extern "C"
 void kernel_polly(pbsize_t  n,
    real L[][n], real x[], real b[]) {

    for (idx_t i = 0; i < n; i++) {
        x[i] = b[i];
        for (idx_t j = 0; j < i; j++) // TODO: make polly skew the loop
            x[i] -= L[i][j] * x[j]; 
        x[i] /= L[i][i];
    }
}
#endif


static void kernel(pbsize_t  n,
                   multarray<real, 2> L, real x[], real b[]) {
#pragma scop
  for (idx_t i = 0; i < n; i++) {
    x[i] = b[i];
    for (idx_t j = 0; j < i; j++)
      x[i] -= L[i][j] * x[j]; 
    x[i] /= L[i][i];
  }
#pragma endscop
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


  auto L = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "L");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true,"x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false,"b");

  for (auto &&_ : state)
    kernel(n, L, x, b);
}


