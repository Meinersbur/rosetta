// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"






__global__ void kernel_A_mul_B(pbsize_t ni, pbsize_t nj, pbsize_t nk,
              real* C,
              real* A,
           real*B) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y;

 
if (i < ni && j < nj) {
    for (idx_t k = 0; k < nk; k++)     
        C[i*nj+j] +=  A[i*nk+k] * B[k*nj+j];
}
}










static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}





static void kernel(pbsize_t ni, pbsize_t nj, pbsize_t nk, pbsize_t nl, pbsize_t nm,
              real* E,
              real* A,
           real*B,
          real*F,
         real* C,
        real* D,
 real* G) {
    unsigned threadsPerBlock = 256;
    dim3 block{threadsPerBlock / 32, 32, 1};

    {
    dim3 grid{num_blocks(ni, block.x), num_blocks(nj, block.y), 1};
    kernel_A_mul_B <<<block ,grid >>> (ni,nj,nk,E,A,B);
    }


    {
    dim3 grid{num_blocks(nj, block.x), num_blocks(nl, block.y), 1};
    kernel_A_mul_B <<<block ,grid >>> (nj,nl,nm,F,C,D);
    }

    {
    dim3 grid{num_blocks(ni, block.x), num_blocks(nl, block.y), 1};
    kernel_A_mul_B <<<block ,grid >>> (ni,nl,nj,G,E,F);
    }
}



void run(State &state, pbsize_t pbsize) {
  pbsize_t ni = pbsize - pbsize / 3;  // 800
  pbsize_t nj = pbsize - pbsize / 4;  // 900
  pbsize_t nk = pbsize - pbsize / 6;  // 1000
  pbsize_t nl = pbsize - pbsize / 12; // 1100
  pbsize_t nm = pbsize;               // 1200



 // auto E = state.allocate_array<real>({ni, nj}, /*fakedata*/ false, /*verify*/ false, "E");
  auto A = state.allocate_array<real>({ni, nk}, /*fakedata*/ true, /*verify*/ false, "A");
  auto B = state.allocate_array<real>({nk, nj}, /*fakedata*/ true, /*verify*/ false, "B");
  //auto F = state.allocate_array<real>({nj, nl}, /*fakedata*/ false, /*verify*/ false, "F");
  auto C = state.allocate_array<real>({nj, nm}, /*fakedata*/ true, /*verify*/ false, "C");
  auto D = state.allocate_array<real>({nm, nl}, /*fakedata*/ true, /*verify*/ false, "D");
  auto G = state.allocate_array<real>({ni, nl}, /*fakedata*/ false, /*verify*/ true, "G");



        real* dev_E  = state.allocate_dev<real>(ni* nj);
          real* dev_A  = state.allocate_dev<real>(ni* nk);
            real* dev_B  = state.allocate_dev<real>(nk* nj);
                            real* dev_F  = state.allocate_dev<real>(nj* nl);
                real* dev_C  = state.allocate_dev<real>(nj* nm);
                    real* dev_D  = state.allocate_dev<real>(nm* nl);
                                  real* dev_G  = state.allocate_dev<real>(ni* nl);


  for (auto &&_ : state) {
               cudaMemset(dev_E, '\0', ni*nj * sizeof(real) );
      cudaMemcpy(dev_A, A.data(),ni*nk* sizeof(real), cudaMemcpyHostToDevice);
        cudaMemcpy(dev_B, B.data(),nk*nj* sizeof(real), cudaMemcpyHostToDevice);
                   cudaMemset(dev_F, '\0', nj*nl * sizeof(real) );
          cudaMemcpy(dev_C, C.data(),nj*nm* sizeof(real), cudaMemcpyHostToDevice);
            cudaMemcpy(dev_D, D.data(),nm*nl* sizeof(real), cudaMemcpyHostToDevice);
                    cudaMemset(dev_F, '\0', ni*nl * sizeof(real) );

    kernel(ni, nj, nk, nl, nm, dev_E, dev_A, dev_B, dev_F, dev_C, dev_D, dev_G);
    

          cudaMemcpy( G.data() ,dev_G,  ni*nl * sizeof(real), cudaMemcpyDeviceToHost ); 

          cudaDeviceSynchronize();
  }

      state.free_dev(dev_E);
        state.free_dev(dev_A);
          state.free_dev(dev_B);
            state.free_dev(dev_F);
              state.free_dev(dev_C);
                          state.free_dev(dev_D);
                                      state.free_dev(dev_G);
}



