// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n, pbsize_t m,
                   real alpha, real beta,
                   multarray<real, 2> C,
                   multarray<real, 2> A) {
  real *pC = &C[0][0];
  real *pA = &A[0][0];


#pragma omp target data map(tofrom                 \
                            : pC [0:n * n]) map(to \
                                                : pA [0:n * m])
  {
#define AccC(i, j) (pC[(i)*n + (j)])
#define AccA(i, j) (pA[(i)*m + (j)])


#pragma omp target teams distribute parallel for collapse(2) default(none) firstprivate(n, m, alpha, beta, pC)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j <= i; j++)
        AccC(i, j) *= beta;

#pragma omp target teams distribute parallel for collapse(2) default(none) firstprivate(n, m, alpha, beta, pC, pA)
    for (idx_t i = 0; i < n; i++)
      for (idx_t j = 0; j <= i; j++)
        for (idx_t k = 0; k < m; k++)
          AccC(i, j) += alpha * AccA(i, k) * AccA(j, k);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");


  for (auto &&_ : state)
    kernel(n, m, alpha, beta, C, A);
}
