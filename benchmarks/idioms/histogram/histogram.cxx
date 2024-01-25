// BUILD: add_benchmark(ppm=serial)
//
// Algorithm: Iterative naive
// Readability: strightforward
// Performance: ILP/SIMD depends on compiler

#include <rosetta.h>


static void kernel(pbsize_t n, uint8_t *data, int32_t *result) {
  for (idx_t i = 0; i < n; i += 1) {
    uint8_t idx = data[i];
    result[idx] += 1;
  }
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<uint8_t>({n}, /*fakedata*/ true, /*verify*/ false, "data");
  auto result = state.allocate_array<int32_t>({256}, /*fakedata*/ false, /*verify*/true, "result");

  for (auto &&_ : state)
    kernel(n, data, result);
}
