// BUILD: add_benchmark(ppm=omp_parallel)
#include "rosetta.h"



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl, pbsize_t nm,
                   multarray<real, 2> E,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   multarray<real, 2> F,
                   multarray<real, 2> C,
                   multarray<real, 2> D,
                   multarray<real, 2> G) {

#pragma omp parallel default(none) firstprivate(ni, nj, nk, nl, nm, E, A, B, F, C, D, G)
  {
/* E := A*B */
#pragma omp for collapse(2) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nj; j++) {
        E[i][j] = 0;
        for (idx_t k = 0; k < nk; ++k)
          E[i][j] += A[i][k] * B[k][j];
      }

/* F := C*D */
#pragma omp for collapse(2) schedule(static)
    for (idx_t i = 0; i < nj; i++)
      for (idx_t j = 0; j < nl; j++) {
        F[i][j] = 0;
        for (idx_t k = 0; k < nm; ++k)
          F[i][j] += C[i][k] * D[k][j];
      }

/* G := E*F */
#pragma omp for collapse(2) schedule(static)
    for (idx_t i = 0; i < ni; i++)
      for (idx_t j = 0; j < nl; j++) {
        G[i][j] = 0;
        for (idx_t k = 0; k < nj; ++k)
          G[i][j] += E[i][k] * F[k][j];
      }
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 6;  // 1000
  pbsize_t nl = pbsize - pbsize / 12; // 1100
  pbsize_t nm = pbsize;               // 1200



  auto E = state.allocate_array<real>({ni, nj}, /*fakedata*/ false, /*verify*/ false, "E");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto F = state.allocate_array<real>({nj, nl}, /*fakedata*/ false, /*verify*/ false, "F");
  auto C = state.allocate_array<real>({nj, nm}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({nm, nl}, /*fakedata*/ true, /*verify*/ false, "D");
  auto G = state.allocate_array<real>({ni, nl}, /*fakedata*/ false, /*verify*/ true, "G");

  for (auto &&_ : state)
    kernel(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
}
