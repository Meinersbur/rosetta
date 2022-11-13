// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"



__global__ void kernel_dev(pbsize_t ni, pbsize_t nj, pbsize_t nk,
    real alpha,
    real beta,
    real* C, real* A, real* B) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y;

    if ( i >= ni || j >= nj) return; 

    C[i*nj+j] *= beta;

    for (idx_t k = 0; k < nk; k++) 
        C[i*nj+j] += alpha * A[i*nk+k] * B[k*nj+j];
}





static
int num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}




static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk,
    real alpha,
    real beta,
    real* C, real* A, real* B) {
        
    int threadsPerBlock = 256;
    dim3 block(threadsPerBlock / 32, 32, 1);
    dim3 grid(num_blocks(ni, block.x), num_blocks(nj, block.y), 1);
    kernel_dev <<<grid, block >>> (ni,nj,nk,alpha,beta,C,A,B);
}


void run(State &state, pbsize_t pbsize) {
    pbsize_t ni = pbsize - pbsize / 4;
    pbsize_t nj = pbsize - pbsize / 8;
    pbsize_t nk = pbsize;

  real alpha = 1.5;
  real beta = 1.2;
  auto C = state.allocate_array<double>({ni, nj}, /*fakedata*/ true, /*verify*/ true);
  auto A = state.allocate_array<double>({ni, nk}, /*fakedata*/ true, /*verify*/ false);
  auto B = state.allocate_array<double>({nk, nj}, /*fakedata*/ true, /*verify*/ false);


  real* dev_C  = state.allocate_dev<real>(ni* nj);
  real* dev_A  = state.allocate_dev<real>(ni* nk);
  real* dev_B  = state.allocate_dev<real>(nk* nj);

  for (auto &&_ : state.manual()) {
          cudaMemcpy(dev_A, A.data(), ni*nk * sizeof(real), cudaMemcpyHostToDevice);
          cudaMemcpy(dev_B, B.data(), nk*nj * sizeof(real), cudaMemcpyHostToDevice);
          cudaMemset(dev_C, 0, ni*nj * sizeof(real) );


          kernel(ni, nj, nk, alpha, beta, dev_C, dev_A, dev_B);


          cudaMemcpy( dev_C, C.data() , ni*nj * sizeof(real), cudaMemcpyDeviceToHost ); 

          cudaDeviceSynchronize();
  }

  state.free_dev(dev_C);
  state.free_dev(dev_A);
  state.free_dev(dev_B);
}
