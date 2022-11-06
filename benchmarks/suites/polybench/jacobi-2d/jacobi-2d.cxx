// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 2> A, multarray<real, 2> B) {
#pragma scop
  for (idx_t t = 0; t < tsteps; t++) {
    for (idx_t i = 1; i < n - 1; i++)
      for (idx_t j = 1; j < n - 1; j++)
        B[i][j] =  (A[i][j] + A[i][j - 1] + A[i][1 + j] + A[1 + i][j] + A[i - 1][j])/5;
    for (idx_t i = 1; i < n - 1; i++)
      for (idx_t j = 1; j < n - 1; j++)
        A[i][j] =  (B[i][j] + B[i][j - 1] + B[i][1 + j] + B[1 + i][j] + B[i - 1][j])/5;
  }
#pragma endscop
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t tsteps = 1; // 500
    pbsize_t n = pbsize; // 1300



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true);
  auto B = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false);


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
