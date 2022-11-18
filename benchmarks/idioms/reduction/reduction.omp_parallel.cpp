// BUILD: add_benchmark(ppm=omp_parallel)

#include "rosetta.h"

static real kernel(pbsize_t n) {
    real sum = 0;
    #pragma omp parallel reduction(+:sum)
    for (idx_t i = 0; i < n; i += 1)
        sum += i;
    return sum;
}

void run(State& state, pbsize_t n) {
 auto sum = state.allocate_array<real>({}, /*fakedata*/ false, /*verify*/ true, "sum");


    for (auto &&_ : state) {1
        sum[0] = kernel(n);
        benchmark::DoNotOptimize(result);
    }
}
