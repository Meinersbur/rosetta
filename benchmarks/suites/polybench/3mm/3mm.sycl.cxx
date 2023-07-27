// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &q,
              pbsize_t nx, pbsize_t ny, pbsize_t nz,
              buffer<real, 1> &X_buf,
              buffer<real, 1> &Y_buf, buffer<real, 1> &Z_buf) {
  q.submit([&](handler &cgh) {
    auto X_acc = X_buf.get_access<access::mode::read>(cgh);
    auto Y_acc = Y_buf.get_access<access::mode::read>(cgh);
    auto Z_acc = Z_buf.get_access<access::mode::write>(cgh);

    cgh.parallel_for<class matmul>(range<2>(nx, ny), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];

      real sum = 0;
      for (idx_t k = 0; k < nz; ++k) {
        sum += X_acc[i * nz + k] * Y_acc[k * ny + j];
      }
      Z_acc[i * ny + j] = sum;
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 6;  // 1000
  pbsize_t nl = pbsize - pbsize / 12; // 1100
  pbsize_t nm = pbsize;               // 1200

  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto C = state.allocate_array<real>({nj, nm}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({nm, nl}, /*fakedata*/ true, /*verify*/ false, "D");
  auto G = state.allocate_array<real>({ni, nl}, /*fakedata*/ false, /*verify*/ true, "G");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(ni * nk));
    buffer<real, 1> B_buf(B.data(), range<1>(nk * nj));
    buffer<real, 1> C_buf(C.data(), range<1>(nj * nm));
    buffer<real, 1> D_buf(D.data(), range<1>(nm * nl));
    buffer<real, 1> E_buf(ni * nj);
    buffer<real, 1> F_buf(nj * nl);
    buffer<real, 1> G_buf(G.data(), range<1>(ni * nl));
    for (auto &&_ : state) {
      mykernel(q, ni, nj, nk, A_buf, B_buf, E_buf);
      mykernel(q, nj, nl, nm, C_buf, D_buf, F_buf);
      mykernel(q, ni, nl, nj, E_buf, F_buf, G_buf);
    }
  }
}
