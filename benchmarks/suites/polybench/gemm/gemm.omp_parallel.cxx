// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk,
                   real alpha,
                   real beta,
                   multarray<real, 2> C, multarray<real, 2> A, multarray<real, 2> B) {
#pragma omp parallel default(none) firstprivate(ni,nj,nk,alpha,beta,C,A,B)
                       {
#pragma omp for collapse(2) schedule(static)
                           for (idx_t i = 0; i < ni; i++) 
                               for (idx_t j = 0; j < nj; j++)
                                   C[i][j] *= beta;

#pragma omp for collapse(2) schedule(static)
                               for (idx_t i = 0; i < ni; i++) 
                                   for (idx_t j = 0; j < nj; j++)
                               for (idx_t k = 0; k < nk; k++) 
                                       C[i][j] += alpha * A[i][k] * B[k][j];
                       }
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t ni = pbsize - pbsize / 4;
    pbsize_t nj = pbsize - pbsize / 8;
    pbsize_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({ni, nj}, /*fakedata*/ true, /*verify*/ true);
  auto A = state.allocate_array<double>({ni, nk}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({nk, nj}, /*fakedata*/ true, /*verify*/ false);

  for (auto &&_ : state)
    kernel(ni, nj, nk, alpha, beta, C, A, B);
}
