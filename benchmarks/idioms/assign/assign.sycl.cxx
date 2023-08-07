// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>


static void kernel_test(cl::sycl::queue q, cl::sycl::buffer<real, 1> buf, pbsize_t n) {

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
  auto pdata = data.data();


  cl::sycl::queue q(cl::sycl::default_selector{});
  {
    cl::sycl::buffer<real, 1> buf(pdata, cl::sycl::range<1>(n));
    for (auto &&_ : state) {
      kernel_test(q, buf, n);
    }
  }
}