// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_stencil(pbsize_t n, real *A, real *B) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y + 1;


  if (i < n - 1 && j < n - 1) {
    B[i * n + j] = (A[i * n + j] + A[i * n + j - 1] + A[i * n + 1 + j] + A[(1 + i) * n + j] + A[(i - 1) * n + j]) / 5;
  }
}



static void kernel(pbsize_t tsteps, pbsize_t n,
                   real *A, real *B) {
  const unsigned int threadsPerBlock = 256;

  for (idx_t t = 1; t <= tsteps; t++) {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n - 2, block.x), num_blocks(n - 2, block.y), 1};
    kernel_stencil<<<grid, block>>>(n, A, B);
    kernel_stencil<<<grid, block>>>(n, B, A);
  }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tsteps = 1; // 500
  pbsize_t n = pbsize; // 1300



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({n, n}, /*fakedata*/ false, /*verify*/ true, "B");

  real *dev_A = state.allocate_dev<real>(n * n);
  real *dev_B = state.allocate_dev<real>(n * n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    kernel(tsteps, n, dev_A, dev_B);
    BENCH_CUDA_TRY(cudaMemcpy(B.data(), dev_B, n * n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_A);
  state.free_dev(dev_B);
}
