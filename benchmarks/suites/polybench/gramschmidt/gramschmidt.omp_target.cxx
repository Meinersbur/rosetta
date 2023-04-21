// BUILD: add_benchmark(ppm=omp_target,sources=[__file__, "gramschmidt-common.cxx"])

#include "gramschmidt-common.h"
#include <rosetta.h>



static real sqr(real v) { return v * v; }

static void kernel(pbsize_t m, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> R, multarray<real, 2> Q) {
  real *pA = &A[0][0];
  real *pR = &R[0][0];
  real *pQ = &Q[0][0];

#pragma omp target data map(tofrom                                 \
                            : pA [0:m * n], pR [0:n * n]) map(from \
                                                              : pQ [0:m * n])
  {

    for (idx_t k = 0; k < n; k++) {
      real sum = 0;

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(k, m, n, pA) reduction(+ \
                                                                                                                                          : sum)
      for (idx_t i = 0; i < m; i++) {
        sum += sqr(pA[i * n + k]);
      }

#pragma omp target // map(to:sum)
      pR[k * n + k] = std::sqrt(sum);


#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(k, m, n, pA, pQ, pR)
      for (int i = 0; i < m; i++)
        pQ[i * n + k] = pA[i * n + k] / pR[k * n + k];


#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(k, m, n, pA, pQ, pR)
      for (idx_t j = k + 1; j < n; j++) {
        pR[k * n + j] = 0;
        for (idx_t i = 0; i < m; i++)
          pR[k * n + j] += pQ[i * n + k] * pA[i * n + j];
      }

#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(k, m, n, pA, pQ, pR)
      for (idx_t j = k + 1; j < n; j++)
        for (idx_t i = 0; i < m; i++)
          pA[i * n + j] -= pQ[i * n + k] * pR[k * n + j];
    }
  }
}



void run(State &state, pbsize_t pbsize) {
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
