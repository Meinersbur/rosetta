// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_mean(pbsize_t m, pbsize_t n,
                            real data[],
                            real cov[],
                            real mean[]) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < m) {
    mean[j] = 0.0;
    for (idx_t i = 0; i < n; i++)
      mean[j] += data[i * m + j];
    mean[j] /= n;
  }
}


__global__ void kernel_reduce(pbsize_t m, pbsize_t n,
                              real data[],
                              real cov[],
                              real mean[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < n && j < m) {
    data[i * m + j] -= mean[j];
  }
}



__global__ void kernel_cov(pbsize_t m, pbsize_t n,
                           real data[],
                           real cov[],
                           real mean[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y + i;



  if (i < m && j < m) {
    cov[i * m + j] = 0.0;
    for (idx_t k = 0; k < n; k++)
      cov[i * m + j] += data[k * m + i] * data[k * m + j];
    cov[i * m + j] /= (n - 1.0);
    cov[j * m + i] = cov[i * m + j];
  }
}


static void kernel(pbsize_t m, pbsize_t n,
                   real data[],
                   real cov[],
                   real mean[]) {
  const unsigned threadsPerBlock = 256;

  kernel_mean<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(m, n, data, cov, mean);


  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(m, block.y), 1};
    kernel_reduce<<<block, grid>>>(m, n, data, cov, mean);
  }


  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(m - 1, block.x), num_blocks(m - 1, block.y), 1};
    kernel_cov<<<block, grid>>>(m, n, data, cov, mean);
  }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 8;

  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto cov = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "cov");

  real *dev_data = state.allocate_dev<real>(n * m);
  real *dev_mean = state.allocate_dev<real>(m);
  real *dev_cov = state.allocate_dev<real>(m * m);

  for (auto &&_ : state) {
    cudaMemcpy(dev_data, data.data(), n * m * sizeof(real), cudaMemcpyHostToDevice);

    kernel(m, n, dev_data, dev_cov, dev_mean);

    cudaMemcpy(mean.data(), dev_mean, m * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(cov.data(), dev_cov, m * m * sizeof(real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
  }

  state.free_dev(dev_data);
  state.free_dev(dev_mean);
  state.free_dev(dev_cov);
}
