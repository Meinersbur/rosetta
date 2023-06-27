// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &myQ, pbsize_t m, pbsize_t n, buffer<real, 1> &A_buf, buffer<real> &s_buf, buffer<real> &q_buf, buffer<real> &p_buf, buffer<real> &r_buf) {
  myQ.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto p_acc = p_buf.get_access<access::mode::read>(cgh);
    auto q_acc = q_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_q>(range<1>(n), [=](id<1> i) {
      q_acc[i] = 0;
      for (idx_t j = 0; j < m; j++) {
       q_acc[i] += A_acc[i * m + j] * p_acc[j];
      }
    });
  });

  myQ.submit([&](handler &cgh) {
    auto A_acc = A_buf.get_access<access::mode::read>(cgh);
    auto r_acc = r_buf.get_access<access::mode::read>(cgh);
    auto s_acc = s_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_s>(range<1>(m), [=](id<1> j) {
      s_acc[j] = 0;
      for (idx_t i = 0; i < n; i++) {
        s_acc[j] += r_acc[i] * A_acc[i * m + j];
      }
    });
  });
  myQ.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t m = pbsize - 19 * pbsize / 21; // 1900
  pbsize_t n = pbsize;                    // 2100

  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto s = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "s");
  auto q = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "q");
  auto p = state.allocate_array<real>({m}, /*fakedata*/ true, /*verify*/ false, "p");
  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");


  queue myQ(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(n * m));
    buffer<real, 1> s_buf(s.data(), range<1>(m));
    buffer<real, 1> p_buf(p.data(), range<1>(m));
    buffer<real, 1> q_buf(q.data(), range<1>(n));
    buffer<real, 1> r_buf(r.data(), range<1>(n));
    for (auto &&_ : state) {
      mykernel(myQ, m, n, A_buf, s_buf, q_buf, p_buf, r_buf);
    }
  }
}