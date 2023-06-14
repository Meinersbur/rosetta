// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>


static void kernel_test(cl::sycl::queue q, pbsize_t n, real data[]) {
  cl::sycl::buffer<real, 1> buf(data, cl::sycl::range<1>(n));
  q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buf.get_access<cl::sycl::access::mode::write>(cgh);
    cgh.parallel_for<class AssignNumbers>(
        cl::sycl::range<1>(n),
        [=](cl::sycl::id<1> id) {
          acc[id] = id;
        });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

  cl::sycl::queue q(cl::sycl::default_selector_v);
  for (auto &&_ : state)
    kernel_test(q, n, data);
}