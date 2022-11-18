// BUILD: add_benchmark(ppm=serial,sources=[__file__,"ludcmp-common.cxx"])

#include <rosetta.h>
#include "ludcmp-common.h"




static void kernel(pbsize_t  n,   multarray<real, 2> A, real b[], real x[], real y[]) {
#pragma scop
  for (idx_t i = 0; i < n; i++) {
    for (idx_t  j = 0; j < i; j++) {
      real w = A[i][j];
      for (idx_t k = 0; k < j; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w / A[j][j];
    }
    for (idx_t j = i; j < n; j++) {
      real w = A[i][j];
      for (idx_t k = 0; k < i; k++) 
        w -= A[i][k] * A[k][j];
      A[i][j] = w;
    }
  }

  for (idx_t i = 0; i < n; i++) {
    real w = b[i];
    for (idx_t j = 0; j < i; j++)
      w -= A[i][j] * y[j];
    y[i] = w;
  }

  for (idx_t i = n - 1; i >= 0; i--) { 
    real w = y[i];
    for (idx_t j = i + 1; j < n; j++)
      w -= A[i][j] * x[j];
    x[i] = w / A[i][i];
  }
#pragma endscop
}





void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize; // 2000


   
  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false, "y");



  for (auto&& _ : state.manual()) {
      ensure_fullrank(n, A);
      {
          auto &&scope = _.scope();
          kernel(n, A, b, x, y);
      }
  }
}
