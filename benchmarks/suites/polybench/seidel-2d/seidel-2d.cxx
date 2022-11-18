// BUILD: add_benchmark(ppm=serial)

#include <rosetta.h>



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 2> A) {
#pragma scop
  for (idx_t t = 0; t <= tsteps - 1; t++)
    for (idx_t i = 1; i <= n - 2; i++)
      for (idx_t j = 1; j <= n - 2; j++)
        A[i][j] = (A[i - 1][j - 1] + A[i - 1][j] + A[i - 1][j + 1] + A[i][j - 1] + A[i][j] + A[i][j + 1] + A[i + 1][j - 1] + A[i + 1][j] + A[i + 1][j + 1]) / 9;
#pragma endscop
}


void run(State &state, int pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");



  for (auto &&_ : state)
    kernel(tsteps, n, A);
}
