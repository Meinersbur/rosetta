// BUILD: add_benchmark(ppm=sycl)
#include <rosetta.h>
// #include <array>
// #include <iostream>
#include <CL/sycl.hpp> 

//using namespace sycl;

static void kernel_test(pbsize_t n, real data[]) {
  cl::sycl::queue q;
  cl::sycl::buffer<real, 1> buf(data, cl::sycl::range<1>(n));

    q.submit([&](cl::sycl::handler& cgh) {
        auto acc = buf.get_access<cl::sycl::access::mode::read_write>(cgh);

        cgh.parallel_for<class AssignNumbers>(
            cl::sycl::range<1>(n),
            [=](cl::sycl::id<1> id) {
                acc[id] = id;
            });
    });
    q.wait_and_throw();
    //std::cout <<"inside kernel: "<< data[5] << std::endl;
}

void run(State &state, pbsize_t n) {
  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");

  for (auto &&_ : state){
    kernel_test(n, data);
  }
}