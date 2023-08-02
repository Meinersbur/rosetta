// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>



static void kernel(pbsize_t tsteps, pbsize_t n,
                   multarray<real, 3> A, multarray<real, 3> B) {
  real *pA = &A[0][0][0];
  real *pB = &B[0][0][0];

#pragma omp target data map(to                           \
                            : pA [0:n * n * n]) map(from \
                                                    : pB [0:n * n * n])
  {

    for (idx_t t = 1; t <= tsteps; t++) {

#pragma omp target teams distribute parallel for collapse(3) dist_schedule(static) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
          for (idx_t k = 1; k < n - 1; k++)
            pB[(i * n + j) * n + k] = (pA[((i + 1) * n + j) * n + k] - 2 * pA[(i * n + j) * n + k] + pA[((i - 1) * n + j) * n + k]) / 8 + (pA[(i * n + (j + 1)) * n + k] - 2 * pA[(i * n + j) * n + k] + pA[(i * n + (j - 1)) * n + k]) / 8 + (pA[(i * n + j) * n + (k + 1)] - 2 * pA[(i * n + j) * n + k] + pA[(i * n + j) * n + (k - 1)]) / 8 + pA[(i * n + j) * n + k];



#pragma omp target teams distribute parallel for collapse(3) dist_schedule(static) schedule(static)
      for (idx_t i = 1; i < n - 1; i++)
        for (idx_t j = 1; j < n - 1; j++)
          for (idx_t k = 1; k < n - 1; k++)
            pA[(i * n + j) * n + k] = (pB[((i + 1) * n + j) * n + k] - 2 * pB[(i * n + j) * n + k] + pB[((i - 1) * n + j) * n + k]) / 8 + (pB[(i * n + (j + 1)) * n + k] - 2 * pB[(i * n + j) * n + k] + pB[(i * n + (j - 1)) * n + k]) / 8 + (pB[(i * n + j) * n + (k + 1)] - 2 * pB[(i * n + j) * n + k] + pB[(i * n + j) * n + (k - 1)]) / 8 + pB[(i * n + j) * n + k];
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