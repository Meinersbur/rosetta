// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


__global__ void kernel_mean(pbsize_t m, pbsize_t n,
                            real *data,
                            real *corr,
                            real mean[],
                            real stddev[]) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j < m) {
    mean[j] = 0.0;
    for (idx_t i = 0; i < n; i++)
      mean[j] += data[i * m + j];
    mean[j] /= n;
  }
}


__global__ void kernel_stddev(pbsize_t m, pbsize_t n,
                              real *data,
                              real *corr,
                              real mean[],
                              real stddev[]) {
  idx_t j = blockDim.x * blockIdx.x + threadIdx.x;
  const real eps = 0.1;

  if (j < m) {
    stddev[j] = 0.0;
    for (idx_t i = 0; i < n; i++)
      stddev[j] += (data[i * m + j] - mean[j]) * (data[i * m + j] - mean[j]);
    stddev[j] /= n;
    stddev[j] = sqrt(stddev[j]);
    /* The following in an inelegant but usual way to handle
       near-zero std. dev. values, which below would cause a zero-
       divide. */
    if (stddev[j] <= eps)
      stddev[j] = 1.0;
  }
}



__global__ void kernel_reduce(pbsize_t m, pbsize_t n,
                              real *data,
                              real *corr,
                              real mean[],
                              real stddev[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < n && j < m) {
    data[i * m + j] -= mean[j];
    data[i * m + j] /= std::sqrt((real)n) * stddev[j];
  }
}



__global__ void kernel_diag(pbsize_t m, pbsize_t n,
                            real *data,
                            real *corr,
                            real mean[],
                            real stddev[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < m) {
    corr[i * m + i] = 1.0;
  }
}



__global__ void kernel_corr(pbsize_t m, pbsize_t n,
                            real *data,
                            real *corr,
                            real mean[],
                            real stddev[]) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y + i + 1;


  //  for (idx_t i = 0; i < m - 1; i++) {
  //       for (idx_t j = i + 1; j < m; j++) {

  if (i < m - 1 && j < m) {
    corr[i * m + j] = 0.0;
    for (int k = 0; k < n; k++)
      corr[i * m + j] += (data[k * m + i] * data[k * m + j]);
    corr[j * m + i] = corr[i * m + j];
  }
}


__global__ void kernel_tail(pbsize_t m, pbsize_t n,
                            real *data,
                            real *corr,
                            real mean[],
                            real stddev[]) {
  corr[(m - 1) * m + m - 1] = 1.0;
}



static void kernel(pbsize_t m, pbsize_t n,
                   real *data,
                   real *corr,
                   real mean[],
                   real stddev[]) {
  const unsigned threadsPerBlock = 256;

  kernel_mean<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(m, n, data, corr, mean, stddev);
  kernel_stddev<<<num_blocks(m, threadsPerBlock), threadsPerBlock>>>(m, n, data, corr, mean, stddev);


  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(m, block.y), 1};
    kernel_reduce<<<block, grid>>>(m, n, data, corr, mean, stddev);
  }



  kernel_diag<<<threadsPerBlock, num_blocks(m, threadsPerBlock)>>>(m, n, data, corr, mean, stddev);



  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(m - 1, block.x), num_blocks(m - 1, block.y), 1};
    kernel_corr<<<block, grid>>>(m, n, data, corr, mean, stddev);
  }


  kernel_tail<<<1, 1>>>(m, n, data, corr, mean, stddev);
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;


  auto data = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "data");
  auto mean = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "mean");
  auto stddev = state.allocate_array<real>({m}, /*fakedata*/ false, /*verify*/ true, "stddev");
  auto corr = state.allocate_array<real>({m, m}, /*fakedata*/ false, /*verify*/ true, "corr");


  real *dev_data = state.allocate_dev<real>(n * m);
  real *dev_corr = state.allocate_dev<real>(m * m);
  real *dev_mean = state.allocate_dev<real>(m);
  real *dev_stddev = state.allocate_dev<real>(m);

  for (auto &&_ : state) {
    cudaMemcpy(dev_data, data.data(), n * m * sizeof(real), cudaMemcpyHostToDevice);
    cudaMemset(dev_corr, '\0', m * m * sizeof(real));
    //      cudaMemset(dev_mean, '\0', m  * sizeof(real));

    kernel(m, n, dev_data, dev_corr, dev_mean, dev_stddev);

    cudaMemcpy(corr.data(), dev_corr, m * m * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(mean.data(), dev_mean, m * sizeof(real), cudaMemcpyDeviceToHost);
    cudaMemcpy(stddev.data(), dev_stddev, m * sizeof(real), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
  }

  state.free_dev(dev_data);
  state.free_dev(dev_corr);
  state.free_dev(dev_mean);
  state.free_dev(dev_stddev);
}
