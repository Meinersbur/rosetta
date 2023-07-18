// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 3> A, multarray<real, 3> B) {
#pragma omp parallel default(none) firstprivate(tsteps, n, A, B)
  {
    for (idx_t t = 1; t <= tsteps; t++) {

#pragma omp for collapse(3) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
          for (idx_t k = 1; k < n - 1; k++) {
            B[i][j][k] = (A[i + 1][j][k] - 2 * A[i][j][k] + A[i - 1][j][k]) / 8 + (A[i][j + 1][k] - 2 * A[i][j][k] + A[i][j - 1][k]) / 8 + (A[i][j][k + 1] - 2 * A[i][j][k] + A[i][j][k - 1]) / 8 + A[i][j][k];
          }



#pragma omp for collapse(3) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
          for (idx_t k = 1; k < n - 1; k++) {
            A[i][j][k] = (B[i + 1][j][k] - 2 * B[i][j][k] + B[i - 1][j][k]) / 8 + (B[i][j + 1][k] - 2 * B[i][j][k] + B[i][j - 1][k]) / 8 + (B[i][j][k + 1] - 2 * B[i][j][k] + B[i][j][k - 1]) / 8 + B[i][j][k];
          }
    }
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 120

  // Linear interpolation of tsteps using the formula
  // Estimated tsteps = tsteps1 + (tsteps2 - tsteps1) * ((pbsize - pbsize1) / (pbsize2 - pbsize1))
  if (pbsize <= 20) {
    tsteps = pbsize * 2;
  }
  else if (pbsize > 20 && pbsize <= 40) {
    tsteps = (3 * pbsize) - 20;
  }
  else if (pbsize > 40 && pbsize <= 120) {
    tsteps = (5 * pbsize) - 100;
  }
  else if (pbsize > 120) {
    tsteps = (6.25 * pbsize) - 250;
  }


  auto A = state.allocate_array<real>({n, n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n, n}, /*fakedata*/ false, /*verify*/ true, "B");


  for (auto &&_ : state)
    kernel(tsteps, n, A, B);
}
