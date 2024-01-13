// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}



__global__ void kernel_x1(pbsize_t n,
                          real x1[],
                          real x2[],
                          real y_1[],
                          real y_2[],
                          real *A) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    for (idx_t j = 0; j < n; j++)
      x1[i] += A[i * n + j] * y_1[j];
  }
}


__global__ void kernel_x2(pbsize_t n,
                          real x1[],
                          real x2[],
                          real y_1[],
                          real y_2[],
                          real *A) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < n) {
    for (idx_t j = 0; j < n; j++)
      x2[i] += A[j * n + i] * y_2[j];
  }
}



static void kernel(pbsize_t n,
                   real x1[],
                   real x2[],
                   real y_1[],
                   real y_2[],
                   real *A) {
  const unsigned threadsPerBlock = 256;


  kernel_x1<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, x1, x2, y_1, y_2, A);
  kernel_x2<<<num_blocks(n, threadsPerBlock), threadsPerBlock>>>(n, x1, x2, y_1, y_2, A);
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize; // 2000



  auto A = state.allocate_array<real>({n, n}, /*fakedata*/ true, /*verify*/ false, "A");
  auto x1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x1");
  auto x2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ true, "x2");
  auto y_1 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_1");
  auto y_2 = state.allocate_array<real>({n}, /*fakedata*/ true, /*verify*/ false, "y_2");


  real *dev_A = state.allocate_dev<real>(n * n);
  real *dev_x1 = state.allocate_dev<real>(n);
  real *dev_x2 = state.allocate_dev<real>(n);
  real *dev_y_1 = state.allocate_dev<real>(n);
  real *dev_y_2 = state.allocate_dev<real>(n);


  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_x1, x1.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_x2, x2.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_y_1, y_1.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_y_2, y_2.data(), n * sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, dev_x1, dev_x2, dev_y_1, dev_y_2, dev_A);
    BENCH_CUDA_TRY(cudaMemcpy(x1.data(), dev_x1, n * sizeof(real), cudaMemcpyDeviceToHost));
    BENCH_CUDA_TRY(cudaMemcpy(x2.data(), dev_x2, n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_A);
  state.free_dev(dev_x1);
  state.free_dev(dev_x2);
  state.free_dev(dev_y_1);
  state.free_dev(dev_y_2);
}
