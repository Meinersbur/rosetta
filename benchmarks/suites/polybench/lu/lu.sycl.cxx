// BUILD: add_benchmark(ppm=sycl,sources=[__file__,"lu-common.cxx"])
#include "lu-common.h"
#include <CL/sycl.hpp>
#include <rosetta.h>

void mykernel(sycl::queue &queue, pbsize_t n, cl::sycl::buffer<real> &A_buf) {

  for (idx_t k = 0; k < n - 1; k++) {
    queue.submit([&](sycl::handler &cgh) {
      auto A_acc = A_buf.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<class kernelU>(
          sycl::range<1>(n - k - 1), [=](sycl::id<1> idx) {
            idx_t i = idx[0] + k + 1;
            A_acc[i * n + k] /= A_acc[k * n + k];
          });
    });
    queue.submit([&](sycl::handler &cgh) {
      auto A_acc = A_buf.get_access<sycl::access::mode::read_write>(cgh);

      cgh.parallel_for<class kernelT>(
          sycl::range<2>(n - k - 1, n - k - 1), [=](sycl::id<2> idx) {
            idx_t i = idx[0] + k + 1;
            idx_t j = idx[1] + k + 1;
            A_acc[i * n + j] -= A_acc[i * n + k] * A_acc[k * n + j];
          });
    });
  }
  queue.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");

  cl::sycl::queue q(cl::sycl::default_selector{});
  //{
  for (auto &&_ : state) {
    ensure_fullrank(n, A);
    {
      cl::sycl::buffer<real, 1> A_buf(A.data(), cl::sycl::range<1>(n * n));
      mykernel(q, n, A_buf);
    }
  }
  //}
}
