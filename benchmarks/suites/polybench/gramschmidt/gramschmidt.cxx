// BUILD: add_benchmark(ppm=serial,sources=[__file__, "gramschmidt-common.cxx"])

#include "gramschmidt-common.h"
#include <rosetta.h>
#include <assert.h>


static real sqr(real v) { return v * v; }

static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> R, multarray<real, 2> Q) {
    assert(n <= m && "Matrix needs sufficient rank");
    A.dump("A");
    R.dump("R");
    Q.dump("Q");

#pragma scop
  for (idx_t k = 0; k < n; k++) {
    R[k][k] = 0;
    for (idx_t i = 0; i < m; i++)
      R[k][k] += sqr(A[i][k]);
    assert(R[k][k] > 0 && "vectors must be linear independent"); // Should also care about not beeing too close to 0 for conditioning
    R[k][k] = std::sqrt(R[k][k]);
    for (idx_t i = 0; i < m; i++)
      Q[i][k] = A[i][k] / R[k][k];
#if 1
    for (idx_t j = k + 1; j < n; j++) {
      R[k][j] = 0;
      for (idx_t i = 0; i < m; i++)
        R[k][j] += Q[i][k] * A[i][j];
      for (idx_t i = 0; i < m; i++)
        A[i][j] -= Q[i][k] * R[k][j];
    }
#endif
  }
#pragma endscop
}



void run(State &state, pbsize_t pbsize) {
    // The original Polybench gramschmidt uses n > m, which necessarily made vectors linearly dependent and therefore orthogonalization ill-defined.   
    pbsize_t m = pbsize;              // 1200
  pbsize_t n = pbsize - pbsize / 6; // 1000



  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
  auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");

  for (auto &&_ : state.manual()) {
    condition(m, n, A);
    {
      auto &&scope = _.scope();
      kernel(m, n, A, R, Q);
    }
  }
}
