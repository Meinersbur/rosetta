// BUILD: add_benchmark(ppm=cuda)

#include <rosetta.h>


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


__global__ void kernel_tmp(pbsize_t  m, pbsize_t  n,
    real alpha, real beta,
real* C,
real* A,
 real* B,real*tmp) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x ;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;


    if (i < m && j < n ) {
        tmp[i*n+j] = 0;
        for (idx_t k = 0; k < i; k++)
            tmp[i*n+j] += B[k*n+j] * A[i*m+k];
    }
}


__global__ void kernel_C(pbsize_t  m, pbsize_t  n,
    real alpha, real beta,
real* C,
real* A,
 real* B,real*tmp) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x ;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;


    if (i < m && j < n )
        C[i*n+j] = beta * C[i*n+j] + alpha * B[i*n+j] * A[i*m+i] + alpha * tmp[i*n+j];
}


__global__ void kernel_sum(pbsize_t  m, pbsize_t  n,
    real alpha, real beta,
real* C,
real* A,
 real* B,real*tmp) {
    idx_t k = blockDim.x * blockIdx.x + threadIdx.x ;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;


    if (k < m-1 && j < n ) {
        for (idx_t i = k + 1; i < m; i++)
          C[k*n+j] += alpha * B[i*n+j] * A[i*m+k];
    }
}




static void kernel(pbsize_t  m, pbsize_t  n,
                   real alpha, real beta,
               real* C,
            real* A,
                real* B,real*tmp) {
                    const  unsigned  int threadsPerBlock = 256;

{
    dim3 block {threadsPerBlock/32, 32,1};
    dim3 grid {num_blocks(m,block.x), num_blocks(n,block.y), 1};
    kernel_tmp<<<block,grid>>>(m,n,alpha,beta,C,A,B,tmp);
}


{
    dim3 block {threadsPerBlock/32, 32,1};
    dim3 grid {num_blocks(m,block.x), num_blocks(n,block.y), 1};
    kernel_C<<<block,grid>>>(m,n,alpha,beta,C,A,B,tmp);
}

// TODO: Combine both kernels?
{
    dim3 block {threadsPerBlock/32, 32,1};
    dim3 grid {num_blocks(m-1,block.x), num_blocks(n,block.y), 1};
    kernel_sum<<<block,grid>>>(m,n,alpha,beta,C,A,B,tmp);
}


                    #if 0
#pragma omp parallel default(none) firstprivate(m,n,alpha,beta,C,A,B,tmp) 
                       {


#pragma omp for collapse(2) schedule (static)
                           for (idx_t i = 0; i < m; i++)
                               for (idx_t j = 0; j < n; j++) {
                                   tmp[i][j] = 0;
                                   for (idx_t k = 0; k < i; k++)
                                       tmp[i][j] += B[k][j] * A[i][k];
                               }

#pragma omp for collapse(2) schedule (static)
                           for (idx_t i = 0; i < m; i++)
                               for (idx_t j = 0; j < n; j++)
                                   C[i][j] = beta * C[i][j] + alpha * B[i][j] * A[i][i] + alpha * tmp[i][j];

#pragma omp for collapse(2) 
for (idx_t k = 0; k < m - 1; k++)
                           for (idx_t j = 0; j < n; j++)
                                   for (idx_t i = k + 1; i < m; i++)
                                       C[k][j] += alpha * B[i][j] * A[i][k];
                       }
                       #endif
}




void run(State &state, pbsize_t pbsize) {
    pbsize_t n = pbsize;
    pbsize_t m = pbsize - pbsize / 8;


  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<real>({m, n}, /*fakedata*/ false, /*verify*/ true,"C");
  auto A = state.allocate_array<real>({m, m}, /*fakedata*/ true, /*verify*/ false,"A");
  auto B = state.allocate_array<real>({m, n}, /*fakedata*/ true, /*verify*/ false, "B");


  real* dev_C= state.allocate_dev<real>(m*n);
  real* dev_A= state.allocate_dev<real>(m*m);
  real* dev_B= state.allocate_dev<real>(m*n);
  real* dev_tmp= state.allocate_dev<real>(m*n);

  for (auto &&_ : state) {
    BENCH_CUDA_TRY( cudaMemcpy(dev_A, A.data(), m*m* sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY( cudaMemcpy(dev_B, B.data(), m*n* sizeof(real), cudaMemcpyHostToDevice));
    kernel(m, n, alpha, beta, dev_C, dev_A, dev_B,dev_tmp);
    BENCH_CUDA_TRY(    cudaMemcpy( C.data() ,dev_C,  m*n*sizeof(real), cudaMemcpyDeviceToHost ));

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }

    state.free_dev(dev_C);
    state.free_dev(dev_A);
    state.free_dev(dev_B);
    state.free_dev(dev_tmp);
}


