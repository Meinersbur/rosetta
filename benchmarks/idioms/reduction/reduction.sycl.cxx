// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>
#include <vector>

static void mykernel(cl::sycl::buffer<real, 1> &res_Buf, cl::sycl::queue q, pbsize_t global_size, pbsize_t local_size, pbsize_t n) {
  pbsize_t length = n;
  idx_t itr = 0;
  idx_t buf_size = global_size / local_size;
  idx_t offset1;
  idx_t offset2;
  do {
    offset1 = pow(local_size, itr);
    offset2 = pow(local_size, (itr + 1));
    q.submit([&](cl::sycl::handler &cgh) {
      auto resAcc = res_Buf.get_access<cl::sycl::access::mode::write>(cgh);
      cl::sycl::accessor<real, 1, cl::sycl::access::mode::read_write, sycl::access::target::local> localSum(cl::sycl::range<1>(local_size), cgh);
      cgh.parallel_for<class reduction_kernel>(
          cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
          [=](cl::sycl::nd_item<1> item) {
            idx_t global_id = item.get_global_id(0);
            idx_t local_id = item.get_local_id(0);
            idx_t group_size = item.get_local_range(0);

            if (global_id >= length) {
              localSum[local_id] = 0;
            }
            if (itr == 0) {
              if (global_id < length) {
                localSum[local_id] = global_id;
              }
            } else {
              if (global_id < length) {
                localSum[local_id] = resAcc[global_id * offset1];
              }
            }
            item.barrier();

            for (idx_t s = group_size / 2; s > 0; s /= 2) {
              if (local_id < s) {
                localSum[local_id] += localSum[local_id + s];
              }
              item.barrier();
            }

            if (local_id == 0) {
              resAcc[item.get_group(0) * offset2] = localSum[0];
            }
          });
    });
    itr += 1;
    length = global_size / local_size;
    global_size = (length / local_size + (length % local_size != 0)) * local_size;
  } while (length > 1);
  q.wait_and_throw();
}

void run(State &state, pbsize_t n) {
  auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");
  multarray<real, 1> sum = sum_owner;

  cl::sycl::queue q(cl::sycl::default_selector_v);
  auto device = q.get_device();

  pbsize_t max_workgroup_size = device.get_info<cl::sycl::info::device::max_work_group_size>();
  pbsize_t local_size = n > max_workgroup_size ? max_workgroup_size / 2 : 32; // We need a power of 2 value for local_size.
  pbsize_t global_size = (n / local_size + (n % local_size != 0)) * local_size;
  {
    cl::sycl::buffer<real, 1> result_buf(global_size / local_size);
    for (auto &&_ : state) {
      mykernel(result_buf, q, global_size, local_size, n);
      auto res_acc = result_buf.get_access<cl::sycl::access::mode::read>(cl::sycl::range<1>(1), cl::sycl::id<1>(0));
      sum[0] = res_acc[0];
    }
  }
}