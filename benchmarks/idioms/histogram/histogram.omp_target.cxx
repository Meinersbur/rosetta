// BUILD: add_benchmark(ppm=omp_target)
//
// Algorithm: Array reduction
// Readability: simple
// Performance: Governed by implementation of array reductions, liky requires a copy of all 256 results per thread.
// 
// Alternatives:
//  * Atomics within a team
//  * Manually combine partial results in second kernel

#include <rosetta.h>


static void kernel(pbsize_t n, uint8_t *data, int32_t *result) {
  #pragma omp target teams distribute parallel for \
    map(to:data[0:n]) map(from:result[0:256]) \
    reduction(+:result[0:256])
  for (idx_t i = 0; i < n; i += 1) {
    uint8_t idx = data[i];
    result[idx] += 1;
  }
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<uint8_t>({n}, /*fakedata*/ true, /*verify*/ false, "data");
  auto result = state.allocate_array<int32_t>({256}, /*fakedata*/ false, /*verify*/true, "result");

  for  (auto &&_ : state)
    kernel(n, data, result);
}
