// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t n,
                     real alpha, real beta,
                     buffer<real, 1> &A_buf,
                     buffer<real, 1> &B_buf,
                     buffer<real, 1> &tmp_buf,
                     buffer<real, 1> &x_buf,
                     buffer<real, 1> &y_buf) {

  q.submit([&](handler &cgh) {
    auto A = A_buf.get_access<access::mode::read>(cgh);
    auto B = B_buf.get_access<access::mode::read>(cgh);
    auto tmp = tmp_buf.get_access<access::mode::read_write>(cgh);
    auto x = x_buf.get_access<access::mode::read>(cgh);
    auto y = y_buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class kernel1>(range<1>(n), [=](id<1> i) {
      tmp[i] = 0.0;
      y[i] = 0.0;
      for (idx_t j = 0; j < n; j++) {
        tmp[i] += A[i * n + j] * x[j];
        y[i] += B[i * n + j] * x[j];
      }
      y[i] = alpha * tmp[i] + beta * y[i];
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ false, "B");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<double>({n}, /*fakedata*/ false, /*verify*/ true, "y");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(n * n));
    buffer<real, 1> B_buf(B.data(), range<1>(n * n));
    buffer<real, 1> tmp_buf(n);
    buffer<real, 1> x_buf(x.data(), range<1>(n));
    buffer<real, 1> y_buf(y.data(), range<1>(n));

    for (auto &&_ : state) {
      mykernel(q, n, alpha, beta, A_buf, B_buf, tmp_buf, x_buf, y_buf);
    }
  }
}