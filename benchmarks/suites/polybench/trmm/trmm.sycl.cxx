// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue q, buffer<real, 1> A_buf, buffer<real, 1> B_buf, pbsize_t n, pbsize_t m, real alpha) {
  q.submit([&](handler &cgh) {
    auto B1 = B_buf.get_access<access::mode::read_write>(cgh);
    auto A1 = A_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel1>(range<1>(n), [=](id<1> j) {
      for (idx_t i = 0; i < m; i++) {
        for (idx_t k = i + 1; k < m; k++) {
          B1[i * n + j] += A1[k * m + i] * B1[k * n + j];
        }
        B1[i * n + j] *= alpha;
      }
    });
  });
  q.wait_and_throw();
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "B");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");

  queue q(default_selector{});
  {
    buffer<real, 1> B_buf(B.data(), range<1>(m * n));
    buffer<real, 1> A_buf(A.data(), range<1>(n * m));
    for (auto &&_ : state) {
      mykernel(q, A_buf, B_buf, n, m, alpha);
    }
  }
}
