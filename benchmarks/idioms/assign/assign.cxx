// BUILD: add_benchmark(ppm=serial)

#include "rosetta.h"

// TODO: use data size instead of passing n separately
static void kernel(pbsize_t n, multarray<real, 1> data) {
    for (idx_t i = 0; i < n; i += 1)
        data[i] = i; // NOT a constant to not allow compiler optimizing to memset.
}

void run(State& state, pbsize_t n) {
    auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

    for (auto &&_ : state)
        kernel(n, data);
}
