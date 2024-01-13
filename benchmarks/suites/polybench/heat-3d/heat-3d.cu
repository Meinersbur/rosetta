// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>



static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_stencil(pbsize_t n, real *A, real *B) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y + 1;
  idx_t k = blockDim.z * blockIdx.z + threadIdx.z + 1;

  if (i < n - 1 && j < n - 1 && k < n - 1) {
    B[(i * n + j) * n + k] = (A[((i + 1) * n + j) * n + k] - 2 * A[(i * n + j) * n + k] + A[((i - 1) * n + j) * n + k]) / 8 + (A[(i * n + (j + 1)) * n + k] - 2 * A[(i * n + j) * n + k] + A[(i * n + (j - 1)) * n + k]) / 8 + (A[(i * n + j) * n + k + 1] - 2 * A[(i * n + j) * n + k] + A[(i * n + j) * n + k - 1]) / 8 + A[(i * n + j) * n + k];
  }
}



static void kernel(pbsize_t tsteps, pbsize_t n, real *A, real *B) {
  const unsigned int threadsPerBlock = 256;

  for (idx_t t = 1; t <= tsteps; t++) {
    dim3 block{1, threadsPerBlock / 32, 32};
    dim3 grid{num_blocks(n - 2, block.x), num_blocks(n - 2, block.y), num_blocks(n - 2, block.z)};
    kernel_stencil<<<grid, block>>>(n, A, B);
    kernel_stencil<<<grid, block>>>(n, B, A);
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 120

  // Linear interpolation of tsteps using the formula
  // Estimated tsteps = tsteps1 + (tsteps2 - tsteps1) * ((pbsize - pbsize1) / (pbsize2 - pbsize1))
  if (pbsize <= 20) {
    tsteps = pbsize * 2;
  } else if (pbsize > 20 && pbsize <= 40) {
    tsteps = (3 * pbsize) - 20;
  } else if (pbsize > 40 && pbsize <= 120) {
    tsteps = (5 * pbsize) - 100;
  } else if (pbsize > 120) {
    tsteps = (6.25 * pbsize) - 250;
  }


  auto A = state.allocate_array<real>({n, n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n, n}, /*fakedata*/ false, /*verify*/ true, "B");

  real *dev_A = state.allocate_dev<real>(n * n * n);
  real *dev_B = state.allocate_dev<real>(n * n * n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), n * n * n * sizeof(real), cudaMemcpyHostToDevice));


    kernel(tsteps, n, dev_A, dev_B);


    BENCH_CUDA_TRY(cudaMemcpy(B.data(), dev_B, n * n * n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }


  state.free_dev(dev_A);
  state.free_dev(dev_B);
}
