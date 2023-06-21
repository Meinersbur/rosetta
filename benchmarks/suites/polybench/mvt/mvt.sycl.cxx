// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t n,
                     buffer<real, 1> &x1_buf,
                     buffer<real, 1> &x2_buf,
                     buffer<real, 1> &y_1_buf,
                     buffer<real, 1> &y_2_buf,
                     buffer<real, 1> &A_buf) {

  q.submit([&](handler &cgh) {
    auto x1 = x1_buf.get_access<access::mode::read_write>(cgh);
    auto y_1 = y_1_buf.get_access<access::mode::read>(cgh);
    auto A = A_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class x1_kernel>(range<1>(n), [=](id<1> i) {
      for (idx_t j = 0; j < n; j++)
        x1[i] += A[i * n + j] * y_1[j];
    });
  });
  q.submit([&](handler &cgh) {
    auto x2 = x2_buf.get_access<access::mode::read_write>(cgh);
    auto y_2 = y_2_buf.get_access<access::mode::read>(cgh);
    auto A = A_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class x2_kernel>(range<1>(n), [=](id<1> i) {
      for (idx_t j = 0; j < n; j++)
        x2[i] += A[j * n + i] * y_2[j];
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x1");
  auto x2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x2");
  auto y_1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_1");
  auto y_2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_2");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(n * n));
    buffer<real, 1> x1_buf(x1.data(), range<1>(n));
    buffer<real, 1> x2_buf(x2.data(), range<1>(n));
    buffer<real, 1> y_1_buf(y_1.data(), range<1>(n));
    buffer<real, 1> y_2_buf(y_2.data(), range<1>(n));
    for (auto &&_ : state) {
      mykernel(q, n, x1_buf, x2_buf, y_1_buf, y_2_buf, A_buf);
    }
  }
}