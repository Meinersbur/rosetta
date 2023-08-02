// BUILD: add_benchmark(ppm=sycl,sources=[__file__, "cholesky-common.cxx"])
#include "cholesky-common.h"
#include <CL/sycl.hpp>
#include <cmath>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &q, pbsize_t n, buffer<real, 1> &A_buf) {
  for (idx_t j = 0; j < n; j++) {
    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task([=]() {
        A_acc[j * n + j] = sqrt(A_acc[j * n + j]);
      });
    });
    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(range<1>(n - j - 1), [=](id<1> idx) {
        idx_t i = idx[0] + j + 1;
        A_acc[i * n + j] /= A_acc[j * n + j];
      });
    });

    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for(range<2>(n - j - 1, n - j - 1), [=](id<2> idx) {
        idx_t i = idx[0] + j + 1;
        idx_t k = idx[1] + j + 1;
        if (k <= i)
          A_acc[i * n + k] -= A_acc[i * n + j] * A_acc[k * n + j];
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(n * n));
    for (auto &&_ : state) {
      ensure_posdefinite(n, A);
      {
        mykernel(q, n, A_buf);
      }
    }
  }
}
