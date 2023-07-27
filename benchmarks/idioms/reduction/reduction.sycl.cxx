// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>
// #include <vector>
#include <sycl/sycl.hpp>

using namespace sycl;

static void mykernel(queue q, buffer<real, 1> &res_Buf, pbsize_t n) {
  q.submit([&](handler &cgh) {
    auto sumr = reduction(res_Buf, cgh, plus<>());
    cgh.parallel_for<class reductionKernel>(nd_range<1>(n, 256), sumr,
                                            [=](nd_item<1> item, auto &sumr_arg) {
                                              idx_t i = item.get_global_id(0);
                                              sumr_arg += i;
                                            });
  });
  q.wait_and_throw();
}

void run(State &state, pbsize_t n) {
  auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");

  queue q(default_selector{});
  {
    buffer<real, 1> sum_buf(sum_owner.data(), range<1>(1));
    for (auto &&_ : state) {
      {
        mykernel(q, sum_buf, n);
      }
    }
  }
}