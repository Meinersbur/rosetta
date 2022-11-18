// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"






__global__ void kernel_A_mul_B(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
                   real alpha, real beta,
                  real *tmp,
                 real * A,
                real * B, real * C, real * D) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y;

 
if (i < ni && j < nj) {
    for (idx_t k = 0; k < nk; k++)     
        tmp[i*nj+j] +=  A[i*nk+k] * B[k*nj+j];
      tmp[i*nj+j] *= alpha ;
}
}



__global__ void kernel_D_plus_tmp_mul_C(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
                   real alpha, real beta,
                  real *tmp,
                 real * A,
                real * B, real * C, real * D) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t l = blockDim.y * blockIdx.y + threadIdx.y;

  
  if (i < ni && l < nl) {
      D[i*nj+l] *=beta;


      for (idx_t j = 0; j < nj; j++)     
          D[i*nl+l] += tmp[i*nj+j] * C[j*nl+l];
  }
}







static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl,
                   real alpha, real beta,
                  real *tmp,
                 real * A,
                real * B, real * C, real * D) {


    unsigned threadsPerBlock = 256;
    dim3 block{threadsPerBlock / 32, 32, 1};

    {
    dim3 grid{num_blocks(ni, block.x), num_blocks(nj, block.y), 1};
    kernel_A_mul_B <<<block ,grid >>> (ni,nj,nk,nl,alpha,beta,tmp,A,B,C,D);
    }


    {
    dim3 grid{num_blocks(ni, block.x), num_blocks(nl, block.y), 1};
      kernel_D_plus_tmp_mul_C <<<block ,grid >>> (ni,nj,nk,nl,alpha,beta,tmp,A,B,C,D);
    }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 12; // 1100
  pbsize_t nl = pbsize;               // 1200

  real alpha = 1.5;
  real beta = 1.2;
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  auto C = state.allocate_array<real>({nj, nl}, /*fakedata*/ true, /*verify*/ false , "C");
  auto D = state.allocate_array<real>({ni, nl}, /*fakedata*/ true, /*verify*/ true, "D");

    real* dev_tmp  = state.allocate_dev<real>(ni* nj);
        real* dev_A  = state.allocate_dev<real>(ni* nk);
            real* dev_B  = state.allocate_dev<real>(nk* nj);
                real* dev_C  = state.allocate_dev<real>(nj* nl);
                    real* dev_D  = state.allocate_dev<real>(ni* nl);

  for (auto &&_ : state) {
                 cudaMemset(dev_tmp, '\0', ni*nj * sizeof(real) );
      cudaMemcpy(dev_A, A.data(),ni*nk* sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, B.data(),nk*nj* sizeof(real), cudaMemcpyHostToDevice);
          cudaMemcpy(dev_C, C.data(),nj*nl* sizeof(real), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_D, D.data(),ni*nl* sizeof(real), cudaMemcpyHostToDevice);
          

    kernel(ni, nj, nk, nl, alpha, beta, dev_tmp, dev_A, dev_B, dev_C, dev_D);
    

          cudaMemcpy( D.data() ,dev_D,  ni*nl * sizeof(real), cudaMemcpyDeviceToHost ); 

          cudaDeviceSynchronize();
  }

      state.free_dev(dev_tmp);
        state.free_dev(dev_A);
          state.free_dev(dev_B);
            state.free_dev(dev_C);
              state.free_dev(dev_D);
}



