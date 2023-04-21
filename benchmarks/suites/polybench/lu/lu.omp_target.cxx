// BUILD: add_benchmark(ppm=omp_target,sources=[__file__,"lu-common.cxx"])

#include "lu-common.h"
#include <rosetta.h>



static void
kernel(pbsize_t n, multarray<real, 2> A) {
  real *pA = &A[0][0];


#pragma omp target data map(tofrom \
                            : pA [0:n * n])
  {
#define AccA(x, y) pA[(x)*n + (y)]

    for (idx_t k = 0; k < n - 1; k++) {


#pragma omp target teams distribute parallel for dist_schedule(static) schedule(static) default(none) firstprivate(n, pA, k)
      for (idx_t i = k + 1; i < n; i++)
        AccA(i, k) /= AccA(k, k);



#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(n, pA, k)
      for (idx_t i = k + 1; i < n; i++)
        for (idx_t j = k + 1; j < n; j++)
          AccA(i, j) -= AccA(i, k) * AccA(k, j);
    }
  }
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
