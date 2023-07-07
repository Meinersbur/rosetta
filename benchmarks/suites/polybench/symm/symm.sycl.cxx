// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void myKernel(pbsize_t m, pbsize_t n, real alpha, real beta,
              buffer<real, 1> &C_buf, buffer<real, 1> &A_buf,
              buffer<real, 1> &B_buf, buffer<real, 1> &tmp_buf, queue &q) {

  q.submit([&](handler &cgh) {
    auto tmp_acc = tmp_buf.get_access<access::mode::read_write>(cgh);
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto B_acc = B_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel_step1>(range<2>(m, n), [=](id<2> idx) {
      auto i = idx[0];
      auto j = idx[1];
      tmp_acc[i * n + j] = 0;
      for (idx_t k = 0; k < i; k++) {
        tmp_acc[i * n + j] += B_acc[k * n + j] * A_acc[i * m + k];
      }
    });
  });

  q.submit([&](handler &cgh) {
    auto C_acc = C_buf.get_access<access::mode::read_write>(cgh);
    auto tmp_acc = tmp_buf.get_access<access::mode::read>(cgh);
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto B_acc = B_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel_step2>(range<2>(m, n), [=](id<2> idx) {
      auto i = idx[0];
      auto j = idx[1];
      C_acc[i * n + j] = beta * C_acc[i * n + j] + alpha * B_acc[i * n + j] * A_acc[i * m + i] + alpha * tmp_acc[i * n + j];
    });
  });

  q.submit([&](handler &cgh) {
    auto C_acc = C_buf.get_access<access::mode::write>(cgh);
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto B_acc = B_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel_step3>(range<2>(n, m - 1), [=](id<2> idx) {
      auto j = idx[0];
      auto k = idx[1];
      for (idx_t i = k + 1; i < m; i++) {
        C_acc[k * n + j] += alpha * B_acc[i * n + j] * A_acc[i * m + k];
      }
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;


  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({m, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "B");

  queue q(default_selector{});
  {
    buffer<real, 1> C_buf(C.data(), range<1>(m * n));
    buffer<real, 1> A_buf(A.data(), range<1>(m * m));
    buffer<real, 1> B_buf(B.data(), range<1>(m * n));
    buffer<real, 1> tmp_buf(m * n);
    for (auto &&_ : state) {
      myKernel(m, n, alpha, beta, C_buf, A_buf, B_buf, tmp_buf, q);
    }
  }
}