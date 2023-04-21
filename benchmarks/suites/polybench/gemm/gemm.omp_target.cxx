// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk,
                   real alpha,
                   real beta,
                   multarray<real, 2> C, multarray<real, 2> A, multarray<real, 2> B) {
  real *pC = &C[0][0];
  real *pA = &A[0][0];
  real *pB = &B[0][0];

#pragma omp target data map(tofrom                                         \
                            : pC [0:ni * nj]) map(to                       \
                                                  : pA [0:ni * nk]) map(to \
                                                                        : pB [0:nk * nj])
  {

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++)
        pC[i * nj + j] *= beta;

#pragma omp target teams distribute parallel for collapse(2) dist_schedule(static) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++)
        for (idx_t k = 0; k < nk; k++)
          pC[i * nj + j] += alpha * pA[i * nk + k] * pB[k * nj + j];
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 4;
  pbsize_t nj = pbsize - pbsize / 8;
  pbsize_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({ni, nj}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");

  for (auto &&_ : state)
    kernel(ni, nj, nk, alpha, beta, C, A, B);
}
