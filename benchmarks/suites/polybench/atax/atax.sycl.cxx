// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, size_t m, size_t n, buffer<real, 1> &A_buf, buffer<real> &x_buf, buffer<real> &y_buf, buffer<real> &tmp_buf) {
  q.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto x_acc = x_buf.get_access<access::mode::read>(cgh);
    auto tmp_acc = tmp_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_tmp>(range<1>(m), [=](id<1> i) {
      real sum = 0;
      for (int j = 0; j < n; ++j) {
        sum += A_acc[i * n + j] * x_acc[j];
      }
      tmp_acc[i] = sum;
    });
  });
  q.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto y_acc = y_buf.get_access<access::mode::read_write>(cgh);
    auto tmp_acc = tmp_buf.get_access<access::mode::read>(cgh);
    cgh.parallel_for<class compute_y>(range<1>(n), [=](id<1> j) {
      real sum = 0;
      for (size_t i = 0; i < m; ++i) {
        sum += A_acc[i * n + j] * tmp_acc[i];
      }
      y_acc[j] = sum;
    });
  });
  q.wait_and_throw();
}

void run(State &state, int pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 10;


  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "x");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(m * n));
    buffer<real, 1> x_buf(x.data(), range<1>(n));
    buffer<real, 1> y_buf(y.data(), range<1>(n));
    buffer<real, 1> tmp_buf(m);
    for (auto &&_ : state) {
      mykernel(q, m, n, A_buf, x_buf, y_buf, tmp_buf);
    }
  }
}