// BUILD: add_benchmark(sycl)
#include "rosetta.h"
#include <CL/sycl.hpp>




static void kernel(int n, double *C, double *B, double *A) {
    cl::sycl::queue q;
    cl::sycl::buffer<double, 1> buf_A(A, cl::sycl::range<1>(n));
	cl::sycl::buffer<double, 1> buf_B(B, cl::sycl::range<1>(n));
	cl::sycl::buffer<double, 1> buf_C(C, cl::sycl::range<1>(n*n));
    q.submit([&](cl::sycl::handler& h) {
		auto a = buf_A.get_access<cl::sycl::access::mode::read>(h);
		auto b = buf_B.get_access<cl::sycl::access::mode::read>(h);
        auto c = buf_C.get_access<cl::sycl::access::mode::read_write>(h);
		//sycl::stream out(1024, 256, h);
		
		h.parallel_for(cl::sycl::range(n, n), [=](cl::sycl::item<2> index) {
			 int i = index.get_id(0);
			 int j = index.get_id(1);
			 c[i * n + j] = a[i] * b[j];
			 //out << i <<":"<<j<<":"<<c[i*n+j] << "=" << a[i] <<"*"<< b[j]<<sycl::endl;
			});
    });
    q.wait_and_throw();
}


static void pairwise_omp_parallel(benchmark::State& state, int n) {
    double *A = static_cast<double*>(new double[n]);
    double *B = static_cast<double*>(new double[n]);
    double *C = static_cast<double*>(new double[n*n]);

    for (auto &&_ : state) {
        kernel(n, C, B, A);
        benchmark::ClobberMemory();
    }

    delete[] A;
    delete[] B;
    delete[] C;
}
