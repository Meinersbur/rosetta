// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(pbsize_t n, real alpha, real beta,
                   multarray<real, 2> A, real *u1, real *v1, real *u2, real *v2, real *w, real *x, real *y, real *z) {
#pragma omp parallel default(none) firstprivate(n,alpha,beta,A,u1,v1,u2,v2,w,x,y,z)
                       {
#pragma omp for collapse(2) schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   A[i][j] +=  u1[i] * v1[j] + u2[i] * v2[j];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   x[i] +=  beta * A[j][i] * y[j];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               x[i] +=  z[i];

#pragma omp for  schedule(static)
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j < n; j++)
                                   w[i] +=  alpha * A[i][j] * x[j] ;
                       }
}


void run(State &state, pbsize_t n) {
      real alpha = 1.5;
  real beta = 1.2;
  auto y = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "y");
  auto z = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "z");
  auto u1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u1");
  auto v1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v1");
  auto u2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u2");
  auto v2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v2");
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto w = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "w");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "x");




  for (auto &&_ : state)
    kernel(n, alpha, beta, A, u1, v1, u2, v2, w, x, y, z);
}
