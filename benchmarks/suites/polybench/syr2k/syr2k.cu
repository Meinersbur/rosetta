// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>

static unsigned num_blocks(int num, int factor) {
  return (num + factor - 1) / factor;
}


__global__ void kernel_beta(pbsize_t n, pbsize_t m,
                            real alpha, real beta,
                            real *C,
                            real *A,
                            real *B) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < n && j <= i)
    C[i * n + j] *= beta;
}

__global__ void kernel_product(pbsize_t n, pbsize_t m,
                               real alpha, real beta,
                               real *C,
                               real *A,
                               real *B) {
  idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
  idx_t j = blockDim.y * blockIdx.y + threadIdx.y;


  if (i < n && j <= i) {
    for (idx_t k = 0; k < m; k++)
      C[i * n + j] += A[j * m + k] * alpha * B[i * m + k] + B[j * m + k] * alpha * A[i * m + k];
  }
}



static void kernel(pbsize_t n, pbsize_t m,
                   real alpha, real beta,
                   real *C,
                   real *A,
                   real *B) {
  const unsigned int threadsPerBlock = 256;

  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(n, block.y), 1};
    kernel_beta<<<block, grid>>>(n, m, alpha, beta, C, A, B);
  }

  {
    dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(n, block.x), num_blocks(n, block.y), 1};
    kernel_product<<<block, grid>>>(n, m, alpha, beta, C, A, B);
  }



#if 0
#pragma omp parallel default(none) firstprivate(n, m, alpha, beta, C, A, B)
                       {
#pragma omp for collapse(2) /* schedule(static) */ 
                           for (idx_t i = 0; i < n; i++) 
                               for (idx_t j = 0; j <= i; j++)
                                   C[i][j] *= beta;

#pragma omp for collapse(2) /* schedule(static) */ 
                           for (idx_t i = 0; i < n; i++)
                               for (idx_t j = 0; j <= i; j++)
                                    for (idx_t k = 0; k < m; k++)                                  
                                        C[i][j] += A[j][k] * alpha * B[i][k] + B[j][k] * alpha * A[i][k];


                       }
#endif
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t n = pbsize;
  pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({n, n}, /*fakedata*/ true, /*verify*/ true, "C");
  auto A = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<double>({n, m}, /*fakedata*/ true, /*verify*/ false, "B");

  real *dev_C = state.allocate_dev<real>(n * n);
  real *dev_A = state.allocate_dev<real>(n * m);
  real *dev_B = state.allocate_dev<real>(n * m);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY(cudaMemcpy(dev_C, C.data(), n * n * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_A, A.data(), n * m * sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY(cudaMemcpy(dev_B, B.data(), n * m * sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, m, alpha, beta, dev_C, dev_A, dev_B);
    BENCH_CUDA_TRY(cudaMemcpy(C.data(), dev_C, n * n * sizeof(real), cudaMemcpyDeviceToHost));

    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_C);
  state.free_dev(dev_A);
  state.free_dev(dev_B);
}
