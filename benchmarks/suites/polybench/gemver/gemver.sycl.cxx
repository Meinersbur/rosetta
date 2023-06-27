// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

static void mykernel(queue &q, pbsize_t n, real alpha, real beta,
                     buffer<real, 1> &A_buf, buffer<real, 1> &u1_buf, buffer<real, 1> &v1_buf,
                     buffer<real, 1> &u2_buf, buffer<real, 1> &v2_buf, buffer<real, 1> &w_buf,
                     buffer<real, 1> &x_buf, buffer<real, 1> &y_buf, buffer<real, 1> &z_buf) {

  q.submit([&](handler &cgh) {
    auto A = A_buf.get_access<access::mode::read_write>(cgh);
    auto u1 = u1_buf.get_access<access::mode::read>(cgh);
    auto v1 = v1_buf.get_access<access::mode::read>(cgh);
    auto u2 = u2_buf.get_access<access::mode::read>(cgh);
    auto v2 = v2_buf.get_access<access::mode::read>(cgh);
    cgh.parallel_for<class update_A>(range<2>(n, n), [=](id<2> idx) {
      idx_t i = idx[0];
      idx_t j = idx[1];
      A[i * n + j] += u1[i] * v1[j] + u2[i] * v2[j];
    });
  });
  q.submit([&](handler &cgh) {
    auto A = A_buf.get_access<access::mode::read>(cgh);
    auto y = y_buf.get_access<access::mode::read>(cgh);
    auto x = x_buf.get_access<access::mode::read_write>(cgh);
    auto z = z_buf.get_access<access::mode::read>(cgh);
    cgh.parallel_for<class update_x>(range<1>(n), [=](id<1> i) {
      for (idx_t j = 0; j < n; j++)
        x[i] += beta * A[j * n + i] * y[j];
      x[i] += z[i];
    });
  });
  q.submit([&](handler &cgh) {
    auto A = A_buf.get_access<access::mode::read>(cgh);
    auto x = x_buf.get_access<access::mode::read>(cgh);
    auto w = w_buf.get_access<access::mode::write>(cgh);
    cgh.parallel_for<class compute_w>(range<1>(n), [=](id<1> i) {
      for (idx_t j = 0; j < n; j++)
        w[i] += alpha * A[i * n + j] * x[j];
    });
  });
}

void run(State &state, pbsize_t n) {
  real alpha = 1.5;
  real beta = 1.2;
  auto y = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "y");
  auto z = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "z");
  auto u1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u1");
  auto v1 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v1");
  auto u2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "u2");
  auto v2 = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ false, "v2");
  auto A = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto w = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "w");
  auto x = state.allocate_array<double>({n}, /*fakedata*/ true, /*verify*/ true, "x");

  queue q(default_selector{});
  {
    buffer<real, 1> y_buf(y.data(), range<1>(n));
    buffer<real, 1> z_buf(z.data(), range<1>(n));
    buffer<real, 1> u1_buf(u1.data(), range<1>(n));
    buffer<real, 1> v1_buf(v1.data(), range<1>(n));
    buffer<real, 1> u2_buf(u2.data(), range<1>(n));
    buffer<real, 1> v2_buf(v2.data(), range<1>(n));
    buffer<real, 1> A_buf(A.data(), range<1>(n * n));
    buffer<real, 1> w_buf(w.data(), range<1>(n));
    buffer<real, 1> x_buf(x.data(), range<1>(n));
    for (auto &&_ : state) {
      mykernel(q, n, alpha, beta, A_buf, u1_buf, v1_buf, u2_buf, v2_buf, w_buf, x_buf, y_buf, z_buf);
    }
  }
}