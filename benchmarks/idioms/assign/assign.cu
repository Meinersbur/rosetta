// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>



__global__ void cuda_assign(pbsize_t n, idx_t i, real *data) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) 
    data[i] = std::sqrt(data[i]);
}

static int num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}





void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; 


  auto data = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true, "data");



  real *dev_data = state.allocate_dev<real>(n);




  for (auto &&_ : state) {
    CUDA_CHECK(   cudaMemcpy(dev_data, data.data(), n * sizeof(real), cudaMemcpyHostToDevice) );

    const int threadsPerBlock = 256;
    cuda_assign<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(n, dev_data);
  
    CUDA_CHECK( cudaMemcpy(data.data(), dev_data, n * sizeof(real), cudaMemcpyDeviceToHost);
    CUDA_CHECK(cudaDeviceSynchronize());
  }


  state.free_dev(dev_data);
}
