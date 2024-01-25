// BUILD: add_benchmark(ppm=omp_parallel)
//
// Algorithm: Global atomic
// Readability: simple
// Performance: No use of locality
// 
// Alternatives:
//  * Use array reduction
//  * Precompute per-thread/NUMA partial results

#include <rosetta.h>


static void kernel(pbsize_t n, uint8_t *data, int32_t *result) {
  #pragma omp parallel for
  for (idx_t i = 0; i < n; i += 1) {
    uint8_t idx = data[i];
    int32_t &ref = result[idx];
    #pragma omp atomic
    ++ref;
  }
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<uint8_t>({n}, /*fakedata*/ true, /*verify*/ false, "data");
  auto result = state.allocate_array<int32_t>({256}, /*fakedata*/ false, /*verify*/true, "result");

  for  (auto &&_ : state)
    kernel(n, data, result);
}
