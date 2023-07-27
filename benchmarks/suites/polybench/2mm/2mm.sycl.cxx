// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &q,
              pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
              real alpha, real beta,
              buffer<real, 1> &tmp_buf,
              buffer<real, 1> &A_buf,
              buffer<real, 1> &B_buf, buffer<real, 1> &C_buf, buffer<real, 1> &D_buf) {
  q.submit([&](handler &cgh) {
    auto tmp_acc = tmp_buf.get_access<access::mode::read_write>(cgh);
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto B_acc = B_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for<class kernel1>(range<2>(ni, nj), [=](id<2> idx) {
      auto i = idx[0];
      auto j = idx[1];
      tmp_acc[i * nj + j] = 0;
      for (idx_t k = 0; k < nk; ++k) {
        tmp_acc[i * nj + j] += alpha * A_acc[i * nk + k] * B_acc[k * nj + j];
      }
    });
  });
  q.submit([&](handler &cgh) {
    auto tmp_acc = tmp_buf.get_access<access::mode::read>(cgh);
    auto C_acc = C_buf.get_access<access::mode::read>(cgh);
    auto D_acc = D_buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for<class kernel2>(range<2>(ni, nl), [=](id<2> idx) {
      auto i = idx[0];
      auto l = idx[1];

      D_acc[i * nl + l] *= beta;
      for (idx_t j = 0; j < nj; ++j) {
        D_acc[i * nl + l] += tmp_acc[i * nj + j] * C_acc[j * nl + l];
      }
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 12; // 1100
  pbsize_t nl = pbsize;               // 1200

  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto C = state.allocate_array<real>({nj, nl}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({ni, nl}, /*fakedata*/ true, /*verify*/ true, "D");

  queue q(default_selector{});
  {
    buffer<real, 1> tmp_buf(ni * nj);
    buffer<real, 1> A_buf(A.data(), range<1>(ni * nk));
    buffer<real, 1> B_buf(B.data(), range<1>(nk * nj));
    buffer<real, 1> C_buf(C.data(), range<1>(nj * nl));
    buffer<real, 1> D_buf(D.data(), range<1>(ni * nl));
    for (auto &&_ : state) {
      mykernel(q, ni, nj, nk, nl, alpha, beta, tmp_buf, A_buf, B_buf, C_buf, D_buf);
    }
  }
}