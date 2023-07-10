// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(pbsize_t n, buffer<real, 1> &path, queue &q) {
  for (idx_t k = 0; k < n; k++) {
    q.submit([&](handler &cgh) {
      auto path_acc = path.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class path_update>(range<2>(n, n), [=](id<2> idx) {
        idx_t i = idx[0];
        idx_t j = idx[1];
        auto newval = path_acc[i * n + k] + path_acc[k * n + j];
        auto &ref = path_acc[i * n + j];
        if (ref > newval)
          ref = newval;
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2800



  auto path = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "path");

  queue q(default_selector{});
  {
    buffer<real> path_buf(path.data(), range<1>(n * n));
    for (auto &&_ : state) {
      mykernel(n, path_buf, q);
    }
  }
}