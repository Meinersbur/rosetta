// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


__global__ void cuda_assign(pbsize_t n, real *data) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n)
    data[i] = i;
}


static int num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;


  auto data = state.allocate_array<real>({n}, /*fakedata*/ false, /*verify*/ true, "data");
  real *dev_data = state.allocate_dev<real>(n);

  for (auto &&_ : state) {
    const int threadsPerBlock = 256;
    cuda_assign<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, dev_data);

    BENCH_CUDA_TRY(cudaMemcpy(data.data(), dev_data, n * sizeof(real), cudaMemcpyDeviceToHost));
    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_data);
}
