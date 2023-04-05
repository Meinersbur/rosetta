// BUILD: add_benchmark(ppm=omp_target,sources=[__file__,"ludcmp-common.cxx"])

#include "ludcmp-common.h"
#include <rosetta.h>


static void kernel(pbsize_t n, multarray<real, 2> A, real b[], real x[], real y[]) {
    real *pA = &A[0][0];

#pragma omp target data map(to:pA[0:n*n],b[0:n]) map(from:x[0:n]) map(alloc:y[0:n])
    {
#define AccA(i,j) pA[(i)*n+(j)]

        for (idx_t i = 0; i < n; i++) {
            for (idx_t j = 0; j < i; j++) {
                real w = AccA(i,j);
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, j, n, pA) schedule(static) reduction(+  : w)
                for (idx_t k = 0; k < j; k++)
                    w -= AccA(i,k) * AccA(k,j);
                AccA(i,j) = w / AccA(j,j);
            }
            for (idx_t j = i; j < n; j++) {
                real w = AccA(i,j);
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, j, n, pA) schedule(static) reduction(+ : w)
                for (idx_t k = 0; k < i; k++)
                    w -= AccA(i,k) * AccA(k,j);
                AccA(i,j)= w;
            }
        }

        for (idx_t i = 0; i < n; i++) {
            real w = b[i];
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, n, pA, y) schedule(static) reduction(+:w)
            for (idx_t j = 0; j < i; j++)
                w -= AccA(i,j) * y[j];
            y[i] = w;
        }

        for (idx_t i = n - 1; i >= 0; i--) {
            real w = y[i];
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, n, pA, x) schedule(static) reduction(+: w)
            for (idx_t j = i + 1; j < n; j++)
                w -= AccA(i,j) * x[j];
            x[i] = w / AccA(i,i);
        }

    }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ false, "y");



  for (auto &&_ : state.manual()) {
    ensure_fullrank(n, A);
    {
      auto &&scope = _.scope();
      kernel(n, A, b, x, y);
    }
  }
}
