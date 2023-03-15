// BUILD: add_benchmark(ppm=omp_target)

#include "rosetta.h"


static void kernel(pbsize_t n, real *A) {
#pragma omp target teams distribute parallel for simd map(from: A[0:n])
  for (idx_t i = 0; i < n; i += 1)
    A[i] += 42;
}

void run(State &state, pbsize_t n) {
  auto A = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "A");

  for (auto &&_ : state)
    kernel(n, A.data());
}
