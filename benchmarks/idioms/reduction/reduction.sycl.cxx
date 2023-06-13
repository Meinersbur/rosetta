// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>
#include <vector>

constexpr pbsize_t local_size = 64; // Work-group size

static real kernel(cl::sycl::queue q, cl::sycl::buffer<real, 1> resultBuffer, pbsize_t global_size, pbsize_t local_size, pbsize_t n) {
  real sum = 0;
  q.submit([&](cl::sycl::handler &cgh) {
    auto resAcc = resultBuffer.get_access<cl::sycl::access::mode::write>(cgh);
    cl::sycl::local_accessor<real, 1> localSum(cl::sycl::range<1>(local_size), cgh);
    
    cgh.parallel_for<class reduction_kernel>(
        cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
        [=](cl::sycl::nd_item<1> item) {
          size_t global_id = item.get_global_id(0);
          size_t local_id = item.get_local_id(0);
          size_t group_size = item.get_local_range(0);

          if (global_id >= n) {
            localSum[local_id] = 0;
          } else {
            localSum[local_id] = global_id;
          }
          item.barrier();

          for (size_t s = group_size / 2; s > 0; s /= 2) {
            if (local_id < s) {
              localSum[local_id] += localSum[local_id + s];
            }
            item.barrier();
          }
          if (local_id == 0) {
            resAcc[0] += localSum[0];
          }
        });
  });
  cl::sycl::host_accessor<real, 1, cl::sycl::access::mode::read> h_result(resultBuffer);
  return (real)h_result[0];
}

void run(State &state, pbsize_t n) {
  auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");
  multarray<real, 1> sum = sum_owner;
  std::vector<real> result(1);

  pbsize_t global_size = n;
  if (n % local_size != 0) {
    global_size = (n / local_size + 1) * local_size;
  }
  cl::sycl::queue q(cl::sycl::default_selector_v);
  {
    cl::sycl::buffer<real, 1> resultBuffer(result.data(), cl::sycl::range<1>(1));

    for (auto &&_ : state) {
      sum[0] = kernel(q, resultBuffer, global_size, local_size, n);
    }
  }
}