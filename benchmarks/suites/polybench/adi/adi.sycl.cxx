// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &queue, pbsize_t tsteps, pbsize_t n, buffer<real> &u_buf, buffer<real> &v_buf, buffer<real> &p_buf, buffer<real> &q_buf) {
  real DX = 1 / (real)n;
  real DY = 1 / (real)n;
  real DT = 1 / (real)tsteps;
  real B1 = 2;
  real B2 = 1;
  real mul1 = B1 * DT / (DX * DX);
  real mul2 = B2 * DT / (DY * DY);

  real a = -mul1 / 2;
  real b = 1 + mul1;
  real c = a;
  real d = -mul2 / 2;
  real e = 1 + mul2;
  real f = d;

  for (idx_t t = 1; t <= tsteps; t++) {
    queue.submit([&](handler &cgh) {
      auto u_acc = u_buf.get_access<access::mode::read_write>(cgh);
      auto v_acc = v_buf.get_access<access::mode::read_write>(cgh);
      auto p_acc = p_buf.get_access<access::mode::read_write>(cgh);
      auto q_acc = q_buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class kernel_col_sweep>(
          range<1>(n - 2), [=](id<1> idx) {
            idx_t i = idx[0] + 1;
            v_acc[0 * n + i] = 1;
            p_acc[i * n + 0] = 0;
            q_acc[i * n + 0] = v_acc[0 * n + i];
            for (idx_t j = 1; j < n - 1; j++) {
              p_acc[i * n + j] = -c / (a * p_acc[i * n + j - 1] + b);
              q_acc[i * n + j] = (-d * u_acc[j * n + i - 1] + (1 + 2 * d) * u_acc[j * n + i] - f * u_acc[j * n + i + 1] - a * q_acc[i * n + j - 1]) / (a * p_acc[i * n + j - 1] + b);
            }
            v_acc[(n - 1) * n + i] = 1;
            for (idx_t j = n - 2; j >= 1; j--)
              v_acc[j * n + i] = p_acc[i * n + j] * v_acc[(j + 1) * n + i] + q_acc[i * n + j];
          });
    });

    queue.submit([&](handler &cgh) {
      auto u_acc = u_buf.get_access<access::mode::read_write>(cgh);
      auto v_acc = v_buf.get_access<access::mode::read_write>(cgh);
      auto p_acc = p_buf.get_access<access::mode::read_write>(cgh);
      auto q_acc = q_buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class kernel_row_sweep>(
          range<1>(n - 2), [=](id<1> idx) {
            idx_t i = idx[0] + 1;
            u_acc[i * n + 0] = 1;
            p_acc[i * n + 0] = 0;
            q_acc[i * n + 0] = u_acc[i * n + 0];
            for (idx_t j = 1; j < n - 1; j++) {
              p_acc[i * n + j] = -f / (d * p_acc[i * n + j - 1] + e);
              q_acc[i * n + j] = (-a * v_acc[(i - 1) * n + j] + (1 + 2 * a) * v_acc[i * n + j] - c * v_acc[(i + 1) * n + j] - d * q_acc[i * n + j - 1]) / (d * p_acc[i * n + j - 1] + e);
            }
            u_acc[i * n + n - 1] = 1;
            for (idx_t j = n - 2; j >= 1; j--)
              u_acc[i * n + j] = p_acc[i * n + j] * u_acc[i * n + j + 1] + q_acc[i * n + j];
          });
    });
  }

  queue.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = pbsize / 2; // 500
  pbsize_t n = pbsize;          // 1000

  auto u = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "u");


  queue q(default_selector{});
  {
    buffer<real, 1> u_buf(u.data(), range<1>(n * n));
    buffer<real, 1> v_buf(n * n);
    buffer<real, 1> p_buf(n * n);
    buffer<real, 1> q_buf(n * n);

    for (auto &&_ : state) {
      mykernel(q, tsteps, n, u_buf, v_buf, p_buf, q_buf);
    }
  }
}
