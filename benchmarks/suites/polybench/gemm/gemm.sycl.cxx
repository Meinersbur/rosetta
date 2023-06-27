// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t ni, pbsize_t nj, pbsize_t nk, real alpha, real beta,
                     buffer<real, 1> &C_buf, buffer<real, 1> &A_buf, buffer<real, 1> &B_buf) {

  q.submit([&](handler &cgh) {
    auto C = C_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class scale_C>(range<2>(ni, nj), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      C[i * nj + j] *= beta;
    });
  });
  q.submit([&](handler &cgh) {
    auto A = A_buf.get_access<access::mode::read>(cgh);
    auto B = B_buf.get_access<access::mode::read>(cgh);
    auto C = C_buf.get_access<access::mode::read_write>(cgh);
    cgh.parallel_for<class matmul_update>(range<2>(ni, nj), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      for (idx_t k = 0; k < nk; k++) {
        C[i * nj + j] += alpha * A[i * nk + k] * B[k * nj + j];
      }
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 4;
  pbsize_t nj = pbsize - pbsize / 8;
  pbsize_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({ni, nj}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");

  queue q(default_selector{});
  {
    buffer<real, 1> C_buf(C.data(), range<1>(ni * nj));
    buffer<real, 1> A_buf(A.data(), range<1>(ni * nk));
    buffer<real, 1> B_buf(B.data(), range<1>(nk * nj));
    for (auto &&_ : state) {
      mykernel(q, ni, nj, nk, alpha, beta, C_buf, A_buf, B_buf);
    }
  }
}
