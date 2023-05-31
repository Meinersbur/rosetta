// BUILD: add_benchmark(ppm=sycl)
#include <rosetta.h>
// #include <array>
// #include <iostream>
#include <sycl/sycl.hpp> //Not required as included in rosetta-common.cpp

using namespace sycl;

// static void kernel(pbsize_t n, real *data, queue Q) {
//     Q.submit([&](handler& h) {
//         accessor A{data, h};
//         h.parallel_for(n, [=](auto& idx) { A[idx] = idx; });
//     });
//     host_accessor A{data};
// }

// void run(State &state, pbsize_t n) {
//   auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");
//   queue Q;
//   buffer B{data};
//   for (auto &&_ : state){   
//     kernel(n, B, Q);
//   }
// }


//void run(State &state, pbsize_t n) {
//   auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");
//   queue Q;
//   buffer B{data};

//   Q.submit([&](handler& h) {
//     accessor A{B, h};
//     h.parallel_for(n, [=](auto& idx) { A[idx] = idx; });
//   });
//   host_accessor A{B};
//   for (int i = 0; i < n; i++)
//     std::cout << "data[" << i << "] = " << A[i] << "\n";

  //return 0;
// }

static void kernel_test(pbsize_t n, real *data) {
  for (idx_t i = 0; i < n; i += 1) {
    // NOT a constant to not allow compiler optimizing to memset.
    data[i] = i;
  }
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

  for (auto &&_ : state)
    kernel_test(n, data);
}