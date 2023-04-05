// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t n, pbsize_t m,
                   real alpha,
                   multarray<real, 2> B, multarray<real, 2> A) {
    real *pB = &B[0][0];
    real *pA = &A[0][0];


#pragma omp target data map(tofrom:pB[0:m*n])  map(to:pA[0:n*m])  
  {
#define AccB(i,j) (pB[(i)*n+(j)])
#define AccA(i,j) (pA[(i)*m+(j)])

#pragma omp target teams distribute parallel for  default(none) firstprivate(n, m, alpha, pB, pA)
    for (idx_t j = 0; j < n; j++)
      for (idx_t i = 0; i < m; i++)
        for (idx_t k = i + 1; k < m; k++)
          AccB(i,j) += AccA(k,i) * AccB(k,j);

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static) default(none) firstprivate(n, m, alpha, pB)
    for (idx_t i = 0; i < m; i++)
      for (idx_t j = 0; j < n; j++)
        AccB(i,j) *= alpha;
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "B");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");

  for (auto &&_ : state)
    kernel(n, m, alpha, B, A);
}
