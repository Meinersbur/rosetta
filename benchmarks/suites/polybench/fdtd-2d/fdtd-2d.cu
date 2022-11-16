// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"


static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}


__global__ void kernel_splat(pbsize_t  tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                  real * ex, real * ey, real * hz, real fict[], idx_t t) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

if (j < ny)
     ey[0*ny+j] = fict[t];
}


__global__ void kernel_ey(pbsize_t  tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                  real * ex, real * ey, real * hz, real fict[], idx_t t) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x+1;
        idx_t j = blockDim.y * blockIdx.y + threadIdx.y;

if (i < nx && j < ny)
           ey[i*ny+j] -= (real)(0.5) * (hz[i*ny+j] - hz[(i - 1)*ny+j]);
}



__global__ void kernel_ex(pbsize_t  tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                  real * ex, real * ey, real * hz, real fict[], idx_t t) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
        idx_t j = blockDim.y * blockIdx.y + threadIdx.y + 1;

if (i < nx && j < ny)
            ex[i*ny+j] -=  (real)(0.5) * (hz[i*ny+j] - hz[i*ny+j - 1]);
}



__global__ void kernel_hz(pbsize_t  tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                  real * ex, real * ey, real * hz, real fict[], idx_t t) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
        idx_t j = blockDim.y * blockIdx.y + threadIdx.y ;

if (i < nx -1 && j < ny-1)
                    hz[i*ny+j] -=  (real)(0.7) * (ex[i*ny+j + 1] - ex[i*ny+j] +  ey[(i + 1)*ny+j] - ey[i*ny+j]);
}




static void kernel(pbsize_t  tmax,
                   pbsize_t nx,
                   pbsize_t ny,
                  real * ex, real * ey, real * hz, real fict[]) {
              const  unsigned threadsPerBlock = 256;

         for (idx_t t = 0; t < tmax; t++) {
            kernel_splat<<<threadsPerBlock,num_blocks(ny,threadsPerBlock)>>>(tmax,nx,ny,ex,ey,hz,fict,t);

{
             dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(nx-1, block.x), num_blocks(ny, block.y), 1};
    kernel_ey <<<block ,grid >>> (tmax,nx,ny,ex,ey,hz,fict,t);
}


{
             dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(nx, block.x), num_blocks(ny-1, block.y), 1};
    kernel_ex <<<block ,grid >>> (tmax,nx,ny,ex,ey,hz,fict,t);
}

{
             dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(nx-1, block.x), num_blocks(ny-1, block.y), 1};
    kernel_hz <<<block ,grid >>> (tmax,nx,ny,ex,ey,hz,fict,t);
}
         }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t tmax = 5 * pbsize / 12;   // 500
  pbsize_t nx = pbsize - pbsize / 6; // 1000
  pbsize_t ny = pbsize;              // 1200



  auto ex = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ex");
  auto ey = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "ey");
  auto hz = state.allocate_array<real>({nx, ny}, /*fakedata*/ true, /*verify*/ true, "hz");
  auto fict = state.allocate_array<real>({tmax}, /*fakedata*/ true, /*verify*/ false, "fict");

  real* dev_ex = state.allocate_dev<real>(nx*ny);
    real* dev_ey = state.allocate_dev<real>(nx*ny);
      real* dev_hz = state.allocate_dev<real>(nx*ny);
        real* dev_fict = state.allocate_dev<real>(tmax);

  for (auto &&_ : state) {
 BENCH_CUDA_TRY( cudaMemcpy(dev_ex, ex.data(),  nx*ny* sizeof(real), cudaMemcpyHostToDevice));
  BENCH_CUDA_TRY( cudaMemcpy(dev_ey, ey.data(),  nx*ny* sizeof(real), cudaMemcpyHostToDevice));
   BENCH_CUDA_TRY( cudaMemcpy(dev_hz, hz.data(),  nx*ny* sizeof(real), cudaMemcpyHostToDevice));
    BENCH_CUDA_TRY( cudaMemcpy(dev_fict, fict.data(),  tmax* sizeof(real), cudaMemcpyHostToDevice));

    kernel(tmax, nx, ny, dev_ex, dev_ey, dev_hz, dev_fict);

      BENCH_CUDA_TRY(    cudaMemcpy( ex.data() ,dev_ex,  nx*ny*sizeof(real), cudaMemcpyDeviceToHost )); 
        BENCH_CUDA_TRY(    cudaMemcpy( ey.data() ,dev_ey,  nx*ny*sizeof(real), cudaMemcpyDeviceToHost )); 
          BENCH_CUDA_TRY(    cudaMemcpy( hz.data() ,dev_hz,  nx*ny*sizeof(real), cudaMemcpyDeviceToHost )); 

    BENCH_CUDA_TRY(  cudaDeviceSynchronize());
  }

                           state.free_dev(dev_ex);     
                                                    state.free_dev(dev_ey);     
                                                                             state.free_dev(dev_hz);     
                                                                                                      state.free_dev(dev_fict);     
}
