// BUILD: add_benchmark(ppm=sycl)
#include <rosetta.h>
#include <sycl/sycl.hpp>

using namespace sycl;

void mykernel(queue q, buffer<real, 1> L_buf, buffer<real, 1> x_buf, buffer<real, 1> b_buf, buffer<real, 1> sum_buf, pbsize_t n) {

  for (idx_t i = 0; i < n; i++) {
    q.submit([&](handler &cgh) {
      auto L_acc = L_buf.get_access<access::mode::read>(cgh);
      auto x_acc = x_buf.get_access<access::mode::read>(cgh);
      auto sumr = reduction(sum_buf, cgh, plus<>());
      cgh.parallel_for<class reductionKernel>(nd_range<1>(i, 1024), sumr,
                                              [=](nd_item<1> item, auto &sumr_arg) {
                                                idx_t j = item.get_global_id(0);
                                                sumr_arg += L_acc[i * n + j] * x_acc[j];
                                              });
    });
    q.submit([&](handler &cgh) {
      auto L_acc = L_buf.get_access<access::mode::read>(cgh);
      auto x_acc = x_buf.get_access<access::mode::read_write>(cgh);
      auto b_acc = b_buf.get_access<access::mode::read>(cgh);
      auto sum_acc = sum_buf.get_access<access::mode::read_write>(cgh);
      cgh.single_task<class Kernel2>([=]() {
        x_acc[i] = (b_acc[i] - sum_acc[0]) / L_acc[i * n + i];
        sum_acc[0] = 0; // Without this the reduction gives wrong result in next iteration
      });
    });
  }
  q.wait_and_throw();
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000


  auto L = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "L");
  auto x = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "x");
  auto b = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "b");

  queue q(default_selector{});
  {
    buffer<real, 1> L_buf(L.data(), range<1>(n * n));
    buffer<real, 1> x_buf(x.data(), range<1>(n));
    buffer<real, 1> b_buf(b.data(), range<1>(n));
    buffer<real, 1> sum_buf(1);
    for (auto &&_ : state) {
      mykernel(q, L_buf, x_buf, b_buf, sum_buf, n);
    }
  }
}