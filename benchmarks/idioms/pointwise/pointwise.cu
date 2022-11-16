// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"



static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}




__global__ void kernel_device(pbsize_t n,                   real A[]) {
        idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
      A[i] += 42;
}





void run(State &state, pbsize_t n) {
    auto A = state.allocate_array<real>({n}, /*fakedata*/true, /*verify*/true, "A");
      real* dev_A = state.allocate_dev<real>(n);

  // TODO: #pragma omp parallel outside of loop
  for (auto &&_ : state) {
 BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(),  n* sizeof(real), cudaMemcpyHostToDevice));

       const  unsigned threadsPerBlock = 256;
    kernel_device <<<threadsPerBlock ,num_blocks(n,threadsPerBlock) >>> (n,dev_A);

     BENCH_CUDA_TRY(  cudaMemcpy( A.data() ,dev_A, n*sizeof(real), cudaMemcpyDeviceToHost )); 

     BENCH_CUDA_TRY( cudaDeviceSynchronize());
  }
}


