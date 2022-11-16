// BUILD: add_benchmark(ppm=omp_target)
#include "rosetta.h"

static void kernel(pbsize_t n, real *A) {
#pragma omp target map(tofrom \
                       : A [0:n])
  for (idx_t i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, pbsize_t n) {
  auto A = state.fakedata_array<real>(n, /*verify*/ true);

  for (auto &&_ : state)
    kernel(n, A.data());
}
