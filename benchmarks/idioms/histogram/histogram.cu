// BUILD: add_benchmark(ppm=cuda)
//
// Algorithm: Device-wide atomic
// Analysis: Better algorithm would be block-wise automic, then combine block results

#include <rosetta.h>


__global__ void cuda_histogram(pbsize_t n, uint8_t *data, int32_t *result) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    uint8_t idx = data[i];
    atomicAdd(&result[idx],1);
  }
}

static int num_blocks(int num, int factor) {
  return  (num + factor - 1) / factor;
}

void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;

  auto data = state.allocate_array<uint8_t>({n}, /*fakedata*/ true, /*verify*/ false, "data");
  auto result = state.allocate_array<int32_t>({256}, /*fakedata*/ false, /*verify*/true, "result");

  uint8_t *dev_data = state.allocate_dev<uint8_t>(n);
  int32_t *dev_result = state.allocate_dev<int32_t>(n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_data, data.data(),  n * sizeof(int32_t), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemset(dev_result, 0, 256 * sizeof(int32_t)));
  
    const int threadsPerBlock = 256;
    cuda_histogram<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n,dev_data, dev_result);

    BENCH_CUDA_TRY(cudaMemcpy(result.data(), dev_result, 256 * sizeof(int32_t), cudaMemcpyDeviceToHost));
    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_data);
  state.free_dev(dev_result);
}
