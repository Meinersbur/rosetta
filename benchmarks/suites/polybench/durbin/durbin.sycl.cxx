// BUILD: add_benchmark(ppm=sycl,sources=[__file__, "durbin-common.cxx"])

#include "durbin-common.h"
#include <CL/sycl.hpp>
#include <rosetta.h>

using namespace cl::sycl;

void mykernel(queue &q, pbsize_t n, buffer<real, 1> &r_buf,
              buffer<real, 1> &y_buf, buffer<real, 1> &z_buf, buffer<real, 1> &sum_buf) {
  real beta = 1;
  buffer<real, 1> beta_buf(&beta, range<1>(1));
  buffer<real, 1> alpha_buf(1);
  q.submit([&](handler &cgh) {
    auto r_acc = r_buf.get_access<access::mode::read>(cgh);
    auto y_acc = y_buf.get_access<access::mode::read_write>(cgh);
    auto alpha_acc = alpha_buf.get_access<access::mode::read_write>(cgh);
    cgh.single_task<class initKernel>([=]() {
      alpha_acc[0] = -r_acc[0];
      y_acc[0] = alpha_acc[0];
    });
  });
  for (idx_t k = 1; k < n; k++) {
    q.submit([&](handler &cgh) {
      auto r_acc = r_buf.get_access<access::mode::read>(cgh);
      auto y_acc = y_buf.get_access<access::mode::read>(cgh);
      auto sumr = reduction(sum_buf, cgh, plus<>());
      cgh.parallel_for<class reductionKernel>(nd_range<1>(k, 256), sumr,
                                              [=](nd_item<1> item, auto &sumr_arg) {
                                                idx_t i = item.get_global_id(0);
                                                sumr_arg += r_acc[k - i - 1] * y_acc[i];
                                              });
    });
    q.submit([&](handler &cgh) {
      auto r_acc = r_buf.get_access<access::mode::read>(cgh);
      auto sum_acc = sum_buf.get_access<access::mode::read>(cgh);
      auto beta_acc = beta_buf.get_access<access::mode::read_write>(cgh);
      auto alpha_acc = alpha_buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class alphaBetaKernel>([=]() {
        beta_acc[0] = (1 - alpha_acc[0] * alpha_acc[0]) * beta_acc[0];
        alpha_acc[0] = -(r_acc[k] + sum_acc[0]) / beta_acc[0];
      });
    });
    q.submit([&](handler &cgh) {
      auto alpha_acc = alpha_buf.get_access<access::mode::read>(cgh);
      auto y_acc = y_buf.get_access<access::mode::read>(cgh);
      auto z_acc = z_buf.get_access<access::mode::read_write>(cgh);

      cgh.parallel_for<class zKernel>(range<1>(k), [=](id<1> i) {
        z_acc[i] = y_acc[i] + alpha_acc[0] * y_acc[k - i - 1];
      });
    });
    q.submit([&](handler &cgh) {
      auto y_acc = y_buf.get_access<access::mode::read_write>(cgh);
      auto z_acc = z_buf.get_access<access::mode::read>(cgh);

      cgh.parallel_for<class yKernel>(range<1>(k), [=](id<1> i) {
        y_acc[i] = z_acc[i];
      });
    });
    q.submit([&](handler &cgh) {
      auto y_acc = y_buf.get_access<access::mode::read_write>(cgh);
      auto sum_acc = sum_buf.get_access<access::mode::read_write>(cgh);
      auto alpha_acc = alpha_buf.get_access<access::mode::read>(cgh);
      cgh.single_task<class endKernel>([=]() {
        y_acc[k] = alpha_acc[0];
        sum_acc[0] = 0;
      });
    });
  }

  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000



  auto r = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "r");
  auto y = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "y");

  initialize_input_vector(n, r);

  queue q(default_selector{});
  {
    buffer<real, 1> r_buf(r.data(), range<1>(n));
    buffer<real, 1> y_buf(y.data(), range<1>(n));
    buffer<real, 1> z_buf(n);

    real sum = 0;
    buffer<real, 1> sum_buf(&sum, range<1>(1));
    for (auto &&_ : state) {
      mykernel(q, n, r_buf, y_buf, z_buf, sum_buf);
    }
  }
}