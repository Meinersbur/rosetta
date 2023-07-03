// BUILD: add_benchmark(ppm=sycl)

#include "rosetta.h"
#include <CL/sycl.hpp>

static void kernel_test(pbsize_t n, real data[]) {
  cl::sycl::queue q;
  cl::sycl::buffer<real, 1> buf(data, cl::sycl::range<1>(n));

  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

    cgh.parallel_for<class Pointwise>(
        cl::sycl::range<1>(n),
        [=](cl::sycl::id<1> id) {
          acc[id] += 42;
        });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t n) {
  auto A = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "A");

  for (auto &&_ : state)
    kernel_test(n, A);
}
