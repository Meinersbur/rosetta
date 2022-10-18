// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"



static void kernel(int ni, int nj, int nk, int nl, int nm,
                   multarray<real, 2> E,
                   multarray<real, 2> A,
                   multarray<real, 2> B,
                   multarray<real, 2> F,
                   multarray<real, 2> C,
                   multarray<real, 2> D,
                   multarray<real, 2> G) {
#pragma scop
  /* E := A*B */
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nj; j++) {
      E[i][j] = 0;
      for (int k = 0; k < nk; ++k)
        E[i][j] += A[i][k] * B[k][j];
    }
  /* F := C*D */
  for (int i = 0; i < nj; i++)
    for (int j = 0; j < nl; j++) {
      F[i][j] = 0;
      for (int k = 0; k < nm; ++k)
        F[i][j] += C[i][k] * D[k][j];
    }
  /* G := E*F */
  for (int i = 0; i < ni; i++)
    for (int j = 0; j < nl; j++) {
      G[i][j] = 0;
      for (int k = 0; k < nj; ++k)
        G[i][j] += E[i][k] * F[k][j];
    }
#pragma endscop
}



void run(State &state, int pbsize) {
  size_t ni = pbsize - pbsize / 3;  // 800
  size_t nj = pbsize - pbsize / 4;  // 900
  size_t nk = pbsize - pbsize / 6;  // 1000
  size_t nl = pbsize - pbsize / 12; // 1100
  size_t nm = pbsize;               // 1200



  auto E = state.allocate_array<double>({ni, nj}, /*fakedata*/ false, /*verify*/ false);
  auto A = state.allocate_array<double>({ni, nk}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({nk, nj}, /*fakedata*/ true, /*verify*/ false);
  auto F = state.allocate_array<double>({nj, nl}, /*fakedata*/ false, /*verify*/ false);
  auto C = state.allocate_array<double>({nj, nm}, /*fakedata*/ true, /*verify*/ false);
  auto D = state.allocate_array<double>({nm, nl}, /*fakedata*/ true, /*verify*/ false);
  auto G = state.allocate_array<double>({ni, nl}, /*fakedata*/ false, /*verify*/ true);

  for (auto &&_ : state)
    kernel(ni, nj, nk, nl, nm, E, A, B, F, C, D, G);
}
