// BUILD: add_benchmark(ppm=omp_task)

#include "rosetta.h"


static real kernel(pbsize_t n) {
    real sum = 0;
    #pragma omp parallel
  {
#pragma omp single
    {
    #pragma omp taskloop
    for (idx_t i = 0; i < n; i += 1) {
        #pragma omp atomic
        sum += i;
    }
    return sum;
    }}
}

void run(State& state, pbsize_t n) {
 auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ true, /*verify*/ true, "sum");
 multarray<real, 1> sum = sum_owner;

    for (auto &&_ : state) {
        sum[0] = kernel(n);
    }
}
