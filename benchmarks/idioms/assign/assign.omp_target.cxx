// BUILD: add_benchmark(ppm=omp_target)

#include <rosetta.h>


static void kernel(pbsize_t n, real data[]) {
#pragma omp target teams distribute parallel for map(from \
                                                     : data [0:n])
  for (idx_t i = 0; i < n; i += 1)
    data[i] = i;
}



void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

  for (auto &&_ : state)
    kernel(n, data);
}
