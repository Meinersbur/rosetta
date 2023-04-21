// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
                   real alpha, real beta,
                   multarray<real, 2> tmp,
                   multarray<real, 2> A,
                   multarray<real, 2> B, multarray<real, 2> C, multarray<real, 2> D) {

  real *Adata = &A[0][0];
  real *Bdata = &B[0][0];
  real *Cdata = &C[0][0];
  real *Ddata = &D[0][0];
  real *tmpdata = &tmp[0][0];

#pragma omp target data map(to                                                         \
                            : Adata [0:ni * nk], Bdata [0:nk * nj], Cdata [0:nj * nl]) \
    map(tofrom                                                                         \
        : Ddata [0:ni * nl])                                                           \
        map(alloc                                                                      \
            : tmpdata [0:ni * nj])
  {

#pragma omp target teams distribute parallel for collapse(2)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++) {
        tmpdata[i * nj + j] = 0;
        for (idx_t k = 0; k < nk; ++k)
          tmpdata[i * nj + j] += alpha * Adata[i * nk + k] * Bdata[k * nj + j];
      }

#pragma omp target teams distribute parallel for collapse(2)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nl; j++) {
        Ddata[i * nl + j] *= beta;
        for (idx_t k = 0; k < nj; ++k)
          Ddata[i * nl + j] += tmpdata[i * nj + k] * Cdata[k * nl + j];
      }
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 12; // 1100
  pbsize_t nl = pbsize;               // 1200

  real alpha = 1.5;
  real beta = 1.2;
  auto tmp = state.allocate_array<real>({ni, nj}, /*fakedata*/ false, /*verify*/ false, "tmp");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto C = state.allocate_array<real>({nj, nl}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({ni, nl}, /*fakedata*/ true, /*verify*/ true, "D");

  for (auto &&_ : state)
    kernel(ni, nj, nk, nl, alpha, beta, tmp, A, B, C, D);
}
