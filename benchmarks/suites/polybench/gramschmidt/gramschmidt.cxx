// BUILD: add_benchmark(ppm=serial,sources=[__file__, "gramschmidt-common.cxx"])

#include <rosetta.h>
#include "gramschmidt-common.h"


static real sqr(real v) { return v*v;}

static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> R, multarray<real, 2> Q) {
#pragma scop
  for (idx_t k = 0; k < n; k++) {
      R[k][k] = 0;
    for (idx_t i = 0; i < m; i++)
        R[k][k] += sqr(A[i][k] );
    R[k][k] =  std::sqrt( R[k][k]);
    for (idx_t i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
#if 1
    for (idx_t j = k + 1; j < n; j++) {
      R[k][j] = 0;
      for (idx_t i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (idx_t i = 0; i < m; i++)
        A[i][j] -=  Q[i][k] * R[k][j];
    }
#endif
  }
#pragma endscop
}




void run(State &state, pbsize_t pbsize) {
    pbsize_t m = pbsize - pbsize / 6; // 1000
    pbsize_t n = pbsize;              // 1200


    auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
    auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
    auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");

    for (auto&& _ : state.manual()) {
        condition(m,n,A);
        {
            auto &&scope = _.scope();
            kernel(m, n, A, R, Q);
        }
    }
}
