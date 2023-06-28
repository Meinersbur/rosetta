// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t nr, pbsize_t nq, pbsize_t np, buffer<real, 1> &A_buf, buffer<real> &C4_buf, buffer<real> &sum_buf) {
  q.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto C4_acc = C4_buf.get_access<access::mode::read>(cgh);
    auto sum_acc = sum_buf.get_access<access::mode::read_write>(cgh);

    cgh.parallel_for(range<3>(nr, nq, np), [=](id<3> idx) {
      idx_t r = idx[0];
      idx_t q = idx[1];
      idx_t p = idx[2];
      real sum = 0;
      for (idx_t s = 0; s < np; s++) {
        sum += A_acc[(r * nq + q) * np + s] * C4_acc[s * np + p];
      }
      sum_acc[(r * nq + q) * np + p] = sum;
    });
  });
  q.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);
    auto sum_acc = sum_buf.get_access<access::mode::read>(cgh);

    cgh.parallel_for(range<3>(nr, nq, np), [=](id<3> idx) {
      idx_t r = idx[0];
      idx_t q = idx[1];
      idx_t p = idx[2];
      idx_t index = (r * nq + q) * np + p;
      A_acc[index] = sum_acc[index];
    });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t nq = pbsize - pbsize / 8;  // 140
  pbsize_t nr = pbsize - pbsize / 16; // 150
  pbsize_t np = pbsize;               // 160

  auto A = state.allocate_array<real>({nr, nq, np}, /*fakedata*/ true, /*verify*/ true, "A");
  auto C4 = state.allocate_array<real>({np, np}, /*fakedata*/ true, /*verify*/ false, "C4");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(nr * nq * np));
    buffer<real, 1> C4_buf(C4.data(), range<1>(np * np));
    buffer<real, 1> sum_buf(nr * nq * np);
    for (auto &&_ : state) {
      mykernel(q, nr, nq, np, A_buf, C4_buf, sum_buf);
    }
  }
}