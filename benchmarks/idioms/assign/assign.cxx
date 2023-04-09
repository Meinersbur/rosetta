// BUILD: add_benchmark(ppm=serial)

#include <rosetta.h>



static void kernel(pbsize_t n, real* data) {
    for (idx_t i = 0; i < n; i += 1) {
        // NOT a constant to not allow compiler optimizing to memset.
        data[i] = i; 
    }
}

void run(State& state, pbsize_t n) {
    auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

    for (auto &&_ : state)
        kernel(n, data);
}
