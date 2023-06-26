// BUILD: add_benchmark(ppm=sycl)
#include <CL/sycl.hpp>
#include <rosetta.h>
#include <vector>

static real mykernel(real* res, real* temp, cl::sycl::queue q, pbsize_t global_size, pbsize_t local_size, pbsize_t n) {
  pbsize_t length = n;
  idx_t itr_flag = 0;
  idx_t buf_size = global_size / local_size;
  do {
    {
      cl::sycl::buffer<real, 1> res_Buf(res, cl::sycl::range<1>(buf_size));
      cl::sycl::buffer<real, 1> temp_Buf(temp, cl::sycl::range<1>(buf_size));

      q.submit([&](cl::sycl::handler &cgh) {
        auto resAcc = res_Buf.get_access<cl::sycl::access::mode::write>(cgh);
        auto global_tmp = temp_Buf.get_access<cl::sycl::access::mode::write>(cgh);
        // cl::sycl::local_accessor<pbsize_t, 1> localSum(cl::sycl::range<1>(local_size), cgh); //Doesn't work with llvm-Intel SYCL
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
              if (itr_flag == 0) {
                if (global_id < length) {
                  localSum[local_id] = global_id;
                }
              } else {
                if (global_id < length) {
                  localSum[local_id] = resAcc[global_id];
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
                global_tmp[item.get_group(0)] = localSum[0];
              }
            });
      });
    }
    std::swap(res, temp);
    itr_flag += 1;
    length = global_size / local_size;
    global_size = (length / local_size + (length % local_size != 0)) * local_size;
  } while (length > 1);

  return res[0];
}

void run(State &state, pbsize_t n) {
  auto sum_owner = state.allocate_array<real>({1}, /*fakedata*/ false, /*verify*/ true, "sum");
  multarray<real, 1> sum = sum_owner;

  cl::sycl::queue q(cl::sycl::default_selector_v);
  auto device = q.get_device();

  pbsize_t max_workgroup_size = device.get_info<cl::sycl::info::device::max_work_group_size>();
  pbsize_t local_size = n > max_workgroup_size ? max_workgroup_size/2 : 32; //We need a power of 2 value for local_size.
  pbsize_t global_size = (n / local_size + (n % local_size != 0)) * local_size;
  auto res = state.allocate_array<real>({global_size / local_size}, /*fakedata*/ false, /*verify*/ false, "res");
  auto temp = state.allocate_array<real>({global_size / local_size}, /*fakedata*/ false, /*verify*/ false, "temp");

    for (auto &&_ : state) {
      sum[0] = mykernel(res, temp, q, global_size, local_size, n);
    }
}