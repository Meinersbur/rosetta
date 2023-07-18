// BUILD: add_benchmark(ppm=sycl,sources=[__file__, "gramschmidt-common.cxx"])

#include "gramschmidt-common.h"
#include <CL/sycl.hpp>
#include <cmath>
#include <rosetta.h>
#include <sycl/sycl.hpp>

using namespace sycl;

void mykernel(queue &q, pbsize_t m, pbsize_t n, buffer<real, 1> &A_buf, buffer<real, 1> &R_buf, buffer<real, 1> &Q_buf, buffer<real, 1> &sum_buf) {
  for (idx_t k = 0; k < n; k++) {
    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read>(cgh);
      auto sumr = reduction(sum_buf, cgh, plus<>());
      cgh.parallel_for<class reductionKernel>(nd_range<1>(m, 256), sumr,
                                              [=](nd_item<1> item, auto &sumr_arg) {
                                                idx_t i = item.get_global_id(0);
                                                sumr_arg += A_acc[i * n + k] * A_acc[i * n + k];
                                              });
    });

    q.submit([&](handler &cgh) {
      auto R_acc = R_buf.get_access<access::mode::read_write>(cgh);
      auto sum_acc = sum_buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class sqrtKernel>([=]() {
        R_acc[k * n + k] = sqrt(sum_acc[0]);
        sum_acc[0] = 0;
      });
    });

    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read>(cgh);
      auto R_acc = R_buf.get_access<access::mode::read>(cgh);
      auto Q_acc = Q_buf.get_access<access::mode::read_write>(cgh);
      cgh.parallel_for<class QKernel>(range<1>(m), [=](id<1> i) {
        Q_acc[i * n + k] = A_acc[i * n + k] / R_acc[k * n + k];
      });
    });

    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read>(cgh);
      auto R_acc = R_buf.get_access<access::mode::read_write>(cgh);
      auto Q_acc = Q_buf.get_access<access::mode::read>(cgh);
      cgh.parallel_for<class RKernel>(range<1>(n - (k + 1)), [=](id<1> idx) {
        idx_t j = idx[0] + k + 1;
        R_acc[k * n + j] = 0;
        for (idx_t i = 0; i < m; i++) {
          R_acc[k * n + j] += Q_acc[i * n + k] * A_acc[i * n + j];
        }
      });
    });

    q.submit([&](handler &cgh) {
      auto A_acc = A_buf.get_access<access::mode::read_write>(cgh);
      auto R_acc = R_buf.get_access<access::mode::read>(cgh);
      auto Q_acc = Q_buf.get_access<access::mode::read>(cgh);
      cgh.parallel_for<class AKernel>(range<1>(n - (k + 1)), [=](id<1> idx) {
        idx_t j = idx[0] + k + 1;
        for (idx_t i = 0; i < m; i++) {
          A_acc[i * n + j] -= Q_acc[i * n + k] * R_acc[k * n + j];
        }
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t m = pbsize;              // 1200
  pbsize_t n = pbsize - pbsize / 6; // 1000


  auto A = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "A");
  auto R = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "R");
  auto Q = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "Q");

  queue q(default_selector{});
  {
    buffer<real, 1> A_buf(A.data(), range<1>(m * n));
    buffer<real, 1> R_buf(R.data(), range<1>(n * n));
    buffer<real, 1> Q_buf(Q.data(), range<1>(m * n));
    real sum = 0;
    buffer<real, 1> sum_buf(&sum, range<1>(1));
    for (auto &&_ : state.manual()) {
      condition(m, n, A);
      {
        auto &&scope = _.scope();

        mykernel(q, m, n, A_buf, R_buf, Q_buf, sum_buf);
      }
    }
  }
}
