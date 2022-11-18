// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


__global__ void kernel_contract(pbsize_t n, pbsize_t m,
    real alpha,
real * B, real* A) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x ;



    if (j <n   ) {
        for (idx_t i = 0; i < m; i++)
          for (idx_t k = i + 1; k < m; k++)
            B[i*n+j] += A[k*m+i] * B[k*n+j];
    }
}

__global__ void kernel_alpha(pbsize_t n, pbsize_t m,
    real alpha,
real * B, real* A) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x ;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;


    if (i <m && j < n  )
        B[i*n+j] *= alpha;
}


static void kernel(pbsize_t n, pbsize_t m,
                   real alpha,
             real * B, real* A) {
                const  unsigned  int threadsPerBlock = 256;


                    kernel_contract<<<threadsPerBlock,num_blocks(n,threadsPerBlock)>>>(n,m,alpha,B,A);


                    {
                        dim3 block {threadsPerBlock/32, 32,1};
                        dim3 grid {num_blocks(m,block.x), num_blocks(n,block.y), 1};
                        kernel_alpha<<<block,grid>>>(n,m,alpha,B,A);
                    }




#if 0
#pragma omp parallel default(none) firstprivate(n,m,alpha,B,A)
                       {
#pragma omp for 
                           for (idx_t j = 0; j < n; j++) 
                                for (idx_t i = 0; i < m; i++)                             
                                   for (idx_t k = i + 1; k < m; k++)
                                       B[i][j] += A[k][i] * B[k][j];

#pragma omp for collapse(2) schedule(static)
                                   for (idx_t i = 0; i < m; i++)
                                       for (idx_t j = 0; j < n; j++)
                                   B[i][j] *= alpha;
                               
                       }
#endif
}



void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 6;

  real alpha = 1.5;
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ true, "B");
  auto A = state.allocate_array<real>({n, m}, /*fakedata*/ true, /*verify*/ false, "A");

  real* dev_B = state.allocate_dev<real>(m*n);
  real* dev_A = state.allocate_dev<real>(n*m);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY( cudaMemcpy(dev_B, B.data(), m*n* sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), n*m* sizeof(real), cudaMemcpyHostToDevice));
    kernel(n, m, alpha, dev_B, dev_A);
    BENCH_CUDA_TRY(    cudaMemcpy( B.data() ,dev_B,  m*n*sizeof(real), cudaMemcpyDeviceToHost ));

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }


    state.free_dev(dev_B);
    state.free_dev(dev_A);
}
