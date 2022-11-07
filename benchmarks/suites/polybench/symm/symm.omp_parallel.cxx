// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


/*
      #pragma omp parallel for
      for (int c0 = 0; c0 < m; c0 += 1)
        #pragma simd
        for (int c1 = 0; c1 < n; c1 += 1)
          Stmt_for_body4(c0, c1);

      #pragma omp parallel for
      for (int c0 = 1; c0 < m; c0 += 1)
        for (int c1 = 0; c1 < n; c1 += 1)
          #pragma minimal dependence distance: 1
          for (int c2 = 0; c2 < c0; c2 += 1)
            Stmt_for_body28(c0, c1, c2);

      #pragma omp parallel for
      for (int c0 = 0; c0 < m; c0 += 1)
        #pragma simd
        for (int c1 = 0; c1 < n; c1 += 1)
          Stmt_for_end44(c0, c1);

      #pragma omp parallel for
      for (int c0 = 0; c0 < n; c0 += 1)
        for (int c1 = 0; c1 < m - 1; c1 += 1)
          #pragma minimal dependence distance: 1
          for (int c2 = c1 + 1; c2 < m; c2 += 1)
            Stmt_for_body10(c2, c0, c1);
*/


static void kernel(pbsize_t  m, pbsize_t  n,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A,
                   multarray<real, 2> B,multarray<real, 2> tmp) {
#pragma omp parallel default(none) firstprivate(m,n,alpha,beta,C,A,B,tmp) 
                       {

#pragma omp for collapse(2) schedule (static)
                           for (idx_t i = 0; i < m; i++)
                               for (idx_t j = 0; j < n; j++) {
                                   tmp[i][j] = 0;
                                   for (idx_t k = 0; k < i; k++)
                                       tmp[i][j] += B[k][j] * A[i][k];
                               }

#pragma omp for collapse(2) schedule (static)
                           for (idx_t i = 0; i < m; i++)
                               for (idx_t j = 0; j < n; j++)
                                   C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * tmp[i][j];

#pragma omp for collapse(2) 
                           for (idx_t j = 0; j < n; j++)
                               for (idx_t k = 0; k < m - 1; k++)
                                   for (idx_t i = k + 1; i < m; i++)
                                       C[k][j] += alpha * B[i][j] * A[i][k];
                       }
}




void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 8;


  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({m, n}, /*fakedata*/ false, /*verify*/ true,"C");
  auto A = state.allocate_array<double>({m, m}, /*fakedata*/ true, /*verify*/ false,"A");
  auto B = state.allocate_array<double>({m, n}, /*fakedata*/ true, /*verify*/ false, "B");
  auto tmp = state.allocate_array<double>({m, n}, /*fakedata*/ true, /*verify*/ false, "tmp");

  for (auto &&_ : state)
    kernel(m, n, alpha, beta, C, A, B,tmp);
}
