// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t m, pbsize_t n,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   multarray<real, 2> tmp) {
    real *pC = &C[0][0];
    real *pA = &A[0][0];
    real *pB = &B[0][0];
    real *ptmp = &tmp[0][0];

#pragma omp target data map(from:pC[0:m*n])  map(to:pA[0:m*m],pB[0:m*n]) map(alloc:ptmp[0:m*n]) 
  {
#define AccC(x,y) (pC[(x)*n+(y)])
#define AccA(x,y) (pA[(x)*m+(y)])
#define AccB(x,y) (pB[(x)*n+(y)])
#define Acctmp(x,y) (ptmp[(x)*n+(y)])

#pragma omp target teams distribute parallel for collapse(2) dist_schedule (static) schedule(static) default(none) firstprivate(m, n, alpha, beta,  pA, pB, ptmp)
    for (idx_t i = 0; i < m; i++)
      for (idx_t j = 0; j < n; j++) {
        Acctmp(i,j) = 0;
        for (idx_t k = 0; k < i; k++)
            Acctmp(i,j) += AccB(k,j) * AccA(i,k);
      }

#pragma omp target teams distribute parallel for  collapse(2) dist_schedule (static)  schedule(static) default(none) firstprivate(m, n, alpha, beta, pC, pA, pB, ptmp)
    for (idx_t i = 0; i < m; i++)
      for (idx_t j = 0; j < n; j++)
          AccC(i,j) = beta * AccC(i,j) + alpha * AccB(i,j) * AccA(i,i) + alpha * Acctmp(i,j);

#pragma omp target teams distribute parallel for  collapse(2) default(none) firstprivate(m, n, alpha, beta, pC, pA, pB, ptmp)
    for (idx_t j = 0; j < n; j++)
      for (idx_t k = 0; k < m - 1; k++)
        for (idx_t i = k + 1; i < m; i++)
            AccC(k,j) += alpha * AccB(i,j) * AccA(i,k);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;


  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({m, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "B");
  auto tmp = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ false);



  for (auto &&_ : state)
    kernel(m, n, alpha, beta, C, A, B, tmp);
}
