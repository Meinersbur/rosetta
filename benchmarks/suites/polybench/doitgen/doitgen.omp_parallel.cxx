// BUILD: add_benchmark(ppm=omp_parallel)

#include <rosetta.h>


static void kernel(pbsize_t nr, pbsize_t nq, pbsize_t np,
                   multarray<real, 3> A, multarray<real, 2> C4, multarray<real, 3> sum) {
#pragma omp parallel for default(none) firstprivate(nr, nq, np, A, C4, sum) collapse(2) schedule(static)
  for (idx_t r = 0; r < nr; r++)
    for (idx_t q = 0; q < nq; q++) {
      for (idx_t p = 0; p < np; p++) {
        sum[r][q][p] = 0;
        for (idx_t s = 0; s < np; s++)
          sum[r][q][p] += A[r][q][s] * C4[s][p];
      }
      for (idx_t p = 0; p < np; p++)
        A[r][q][p] = sum[r][q][p];
    }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t nq = pbsize - pbsize / 8;  // 140
  pbsize_t nr = pbsize - pbsize / 16; // 150
  pbsize_t np = pbsize;               // 160


  auto A = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ true, /*verify*/ true, "A");
  auto C4 = state.allocate_array<real>({np, np}, /*fakedata*/ true, /*verify*/ false, "C4");
  auto sum = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ false, /*verify*/ false, "sum");

  for (auto &&_ : state)
    kernel(nr, nq, np, A, C4, sum);
}
