// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"


static void kernel(pbsize_t n, multarray<real, 1> data) {
    #pragma omp parallel for
    for (idx_t i = 0; i < n; i += 1)
        data[i] = i; 
}

void run(State& state, pbsize_t n) {
    auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

    for (auto &&_ : state)
        kernel(n, data);
}
