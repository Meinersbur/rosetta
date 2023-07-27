// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void func(handler &cgh, pbsize_t tsteps, pbsize_t n, buffer<real, 1> &A_buf, buffer<real, 1> &B_buf) {
  auto A_acc = A_buf.get_access<access::mode::read>(cgh);
  auto B_acc = B_buf.get_access<access::mode::write>(cgh);

  cgh.parallel_for<class kernel_step1>(range<2>(n - 2, n - 2), [=](id<2> idx) {
    auto i = idx[0] + 1;
    auto j = idx[1] + 1;
    B_acc[i * n + j] = (A_acc[i * n + j] + A_acc[i * n + j - 1] + A_acc[i * n + 1 + j] + A_acc[(1 + i) * n + j] + A_acc[(i - 1) * n + j]) / 5;
  });
}

void mykernel(queue &q, pbsize_t tsteps, pbsize_t n, buffer<real, 1> &A_buf, buffer<real, 1> &B_buf) {
  for (pbsize_t t = 0; t < tsteps; t++) {
    q.submit([&](handler &cgh) { func(cgh, tsteps, n, A_buf, B_buf); });
    q.submit([&](handler &cgh) { func(cgh, tsteps, n, B_buf, A_buf); });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 1300

  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "B");

  queue q(default_selector{});
  {
    buffer<real> A_buf(A.data(), range<1>(n * n));
    buffer<real> B_buf(B.data(), range<1>(n * n));
    for (auto &&_ : state) {
      mykernel(q, tsteps, n, A_buf, B_buf);
    }
  }
}