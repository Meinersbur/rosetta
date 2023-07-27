// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void kernel_stencil(pbsize_t n, buffer<real, 1> &A_buf, buffer<real, 1> &B_buf, queue &q) {
  q.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto B_acc = B_buf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class stencil>(range<3>(n - 2, n - 2, n - 2), [=](id<3> id) {
      idx_t i = id[0] + 1;
      idx_t j = id[1] + 1;
      idx_t k = id[2] + 1;
      B_acc[(i * n + j) * n + k] = (A_acc[((i + 1) * n + j) * n + k] - 2 * A_acc[(i * n + j) * n + k] + A_acc[((i - 1) * n + j) * n + k]) / 8 + (A_acc[(i * n + (j + 1)) * n + k] - 2 * A_acc[(i * n + j) * n + k] + A_acc[(i * n + (j - 1)) * n + k]) / 8 + (A_acc[(i * n + j) * n + k + 1] - 2 * A_acc[(i * n + j) * n + k] + A_acc[(i * n + j) * n + k - 1]) / 8 + A_acc[(i * n + j) * n + k];
    });
  });
}

void mykernel(pbsize_t tsteps, pbsize_t n, buffer<real, 1> &A_buf, buffer<real, 1> &B_buf, queue &q) {
  for (idx_t t = 1; t <= tsteps; t++) {
    kernel_stencil(n, A_buf, B_buf, q);
    kernel_stencil(n, B_buf, A_buf, q);
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 120



  auto A = state.allocate_array<real>({n, n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n, n}, /*fakedata*/ false, /*verify*/ true, "B");

  queue q(default_selector{});
  {
    buffer<real> A_buf(A.data(), range<1>(n * n * n));
    buffer<real> B_buf(B.data(), range<1>(n * n * n));
    for (auto &&_ : state) {
      mykernel(tsteps, n, A_buf, B_buf, q);
    }
  }
}