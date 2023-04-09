// BUILD: add_benchmark(ppm=omp_target,sources=[__file__,"ludcmp-common.cxx"])

#include "ludcmp-common.h"
#include <rosetta.h>

#include <unistd.h>

static void kernel(pbsize_t n, multarray<real, 2> A, real b[], real x[], real y[]) {
    real *pA = &A[0][0];
    real w = 0;

#pragma omp target data map(to:pA[0:n*n],b[0:n]) map(from:x[0:n]) map(alloc:y[0:n])  map(alloc:w)
    {
#define AccA(i,j) (pA[(i)*n+(j)])
        int k = 0;
        for (int l = 0; l < 4; l += 1) {
        for (idx_t i = 0; i < n; i++) {

           
                fprintf(stderr, "a l=%d i=%d, k=%d\n", l, i,k);
                for (idx_t j = 0; j < i; j++) {
                    real w1 = 0;
#pragma omp target 
                    w = 0;
                    // k+=1;


//#pragma omp target teams distribute parallel  for  dist_schedule(static) firstprivate(i, j, n, pA) schedule(static) reduction(+  : w)   default(none)  defaultmap(none) 
//                    for (idx_t k = 0; k < j; k++)
//                        w -= AccA(i, k) * AccA(k, j);
// k+=1;
#pragma omp target
                   AccA(i, j) = (AccA(i, j) + w) / AccA(j, j);
                   k+=1;
                }
            }
       
#if 0
fprintf(stderr, "b i=%d\n",i);
            for (idx_t j = i; j < n; j++) {
                real w2 = 0;
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, j, n, pA) schedule(static) reduction(+ : w2) map(tofrom:w2) defaultmap(none)
                for (idx_t k = 0; k < i; k++) 
                      w2 -= AccA(i, k) * AccA(k, j);
#pragma omp target  map(to:w2)
                AccA(i,j) += w2;
            }
#endif
        }

#if 0
        for (idx_t i = 0; i < n; i++) {
            fprintf(stderr, "c i=%d\n",i);

            real w3 = 0;
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, n, pA, y) schedule(static) reduction(+:w3) map(tofrom:w3) defaultmap(none)
            for (idx_t j = 0; j < i; j++) 
                    w3 -= AccA(i, j) * y[j];
#pragma omp target map(to:w3,i) map(tofrom:y[0:n],b[0:n])  defaultmap(none)
            y[i] = b[i] + w3;
        }

  
        for (idx_t i = n - 1; i >= 0; i--) {
            fprintf(stderr, "d i=%d\n",i);

            real w4 = 0;
#pragma omp target teams distribute parallel  for  dist_schedule(static)  default(none) firstprivate(i, n, pA, x) schedule(static) reduction(+: w4) map(tofrom:w4)
            for (idx_t j = i + 1; j < n; j++) 
                    w4 -= AccA(i, j) * x[j];
#pragma omp target map(to:w4)
            x[i] = (y[i] + w4) / AccA(i,i);
        }
#endif 

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
