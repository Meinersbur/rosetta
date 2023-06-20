// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>
#include <vector>

static real kernel(cl::sycl::queue q, cl::sycl::buffer<pbsize_t, 1> resultBuffer, pbsize_t global_size, pbsize_t local_size, pbsize_t n) {
  pbsize_t sum = 0;
  pbsize_t length = n;
  do {
    q.submit([&](cl::sycl::handler &cgh) {
      auto resAcc = resultBuffer.get_access<cl::sycl::access::mode::write>(cgh);
      //cl::sycl::local_accessor<pbsize_t, 1> localSum(cl::sycl::range<1>(local_size), cgh); //Doesn't work with llvm-Intel SYCL
      cl::sycl::accessor<pbsize_t, 1, cl::sycl::access::mode::read_write, sycl::access::target::local> localSum(cl::sycl::range<1>(local_size), cgh);
      cgh.parallel_for<class reduction_kernel>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
          [=](cl::sycl::nd_item<1> item) {
            idx_t global_id = item.get_global_id(0);
            idx_t local_id = item.get_local_id(0);
            idx_t group_size = item.get_local_range(0);

            if (global_id >= length) {
              localSum[local_id] = 0;
            } else {
              localSum[local_id] = resAcc[global_id];
            }
            item.barrier();

            for (idx_t s = group_size / 2; s > 0; s /= 2) {
              if (local_id < s) {
                localSum[local_id] += localSum[local_id + s];
              }
              item.barrier();
            }

            if (local_id == 0) {
              resAcc[item.get_group(0)] = localSum[0];
            }
          });
    });
    q.wait();
    length = global_size / local_size;
    global_size = length;
    if (length % local_size != 0) {
      global_size = (length / local_size + 1) * local_size;
    }
  } while (length > 1);
  
    cl::sycl::host_accessor<pbsize_t, 1, cl::sycl::access::mode::read> h_result(resultBuffer);
    sum = h_result[0];
  
  return sum;
}

void run(State &state, pbsize_t n) {
  auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");
  multarray<real, 1> sum = sum_owner;
  std::vector<pbsize_t> result(n);
  for (idx_t i = 0; i < n; i++) {
    result[i] = i;
  }

  cl::sycl::queue q(cl::sycl::default_selector_v);
  auto device = q.get_device();

  int max_workgroup_size = device.get_info<cl::sycl::info::device::max_work_group_size>();
  pbsize_t local_size = n > max_workgroup_size ? max_workgroup_size : n;
  pbsize_t global_size = n;
  if (n % local_size != 0) {
    global_size = (n / local_size + 1) * local_size;
  }



  {
    cl::sycl::buffer<pbsize_t, 1> resultBuffer(result.data(), cl::sycl::range<1>(n));
    for (auto &&_ : state) {
      sum[0] = kernel(q, resultBuffer, global_size, local_size, n);
    }
  }
}