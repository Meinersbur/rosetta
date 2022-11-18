// BUILD: add_benchmark(ppm=cuda)

#include "rosetta.h"

static
unsigned num_blocks(int num, int factor) {
    return (num + factor -1)/factor ;
}



__global__ void kernel_y1_rowsweep(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2,
          real a1,
          real a2,
          real b1,
          real b2
          ) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i< w) {
             real ym1 = 0;
                               real ym2 = 0;
                               real xm1 = 0;
                               for (idx_t j = 0; j < h; j++) {
                                   y1[i*h+j] = a1 * imgIn[i*h+j] + a2 * xm1 + b1 * ym1 + b2 * ym2;
                                   xm1 = imgIn[i*h+j];
                                   ym2 = ym1;
                                   ym1 = y1[i*h+j];
                               }
  }
}

__global__ void kernel_y2_rowsweep(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2,
          real a3,
          real a4,
          real b1,
          real b2
          ) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i< w) {
                               real yp1 = 0;
                               real yp2 = 0;
                               real xp1 = 0;
                               real xp2 = 0;
                               for (idx_t j = h - 1; j >= 0; j--) {
                                   y2[i*h+j] = a3 * xp1 + a4 * xp2 + b1 * yp1 + b2 * yp2;
                                   xp2 = xp1;
                                   xp1 = imgIn[i*h+j];
                                   yp2 = yp1;
                                   yp1 = y2[i*h+j];
                               }
  }
}


__global__ void kernel_out(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2, real c) {
    idx_t i = blockDim.x * blockIdx.x + threadIdx.x;
    idx_t j = blockDim.y * blockIdx.y + threadIdx.y;



 
if (i < w && j < h) {
                                   imgOut[i*h+j] = c * (y1[i*h+j] + y2[i*h+j]);
}
}


__global__ void kernel_y1_colsweep(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2,
          real a5,
          real a6,
          real b1,
          real b2
          ) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j< h) {
                               real tm1 = 0;
                               real ym1 = 0;
                               real ym2 = 0;
                               for (idx_t i = 0; i < w; i++) {
                                   y1[i*h+j] = a5 * imgOut[i*h+j] + a6 * tm1 + b1 * ym1 + b2 * ym2 ;
                                   tm1 = imgOut[i*h+j];
                                   ym2 = ym1;
                                   ym1 = y1[i*h+j];
                               }
  }
}


__global__ void kernel_y2_colsweep(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2,
          real a7,
          real a8,
          real b1,
          real b2
          ) {
    idx_t j = blockDim.x * blockIdx.x + threadIdx.x;

  if (j< h) {
                               real tp1 = 0;
                               real tp2 = 0;
                               real yp1 = 0;
                               real yp2 = 0;
                               for (idx_t i = w - 1; i >= 0; i--) {
                                   y2[i*h+j] = a7 * tp1 + a8 * tp2 + b1 * yp1 + b2 * yp2;
                                   tp2 = tp1;
                                   tp1 = imgOut[i*h+j];
                                   yp2 = yp1;
                                   yp1 = y2[i*h+j];
                               }
  }
}



static void kernel(pbsize_t w, pbsize_t h,
                   real alpha,
                  real* imgIn,
                real*imgOut,
                real*y1,
          real* y2) {
    real k = (1 - std::exp(-alpha)) * (1 - std::exp(-alpha)) / (1 + 2 * alpha * std::exp(-alpha) - std::exp(2 * alpha));
    real a1 = k;
        real a5 = k;
    real a6 = k * std::exp(-alpha) * (alpha - 1);
    real a2 = a6;
    real a7 = k * std::exp(-alpha) * (alpha + 1);
    real a3 = a7;
    real a8 = -k * std::exp(-2 * alpha);
    real a4 = a8;
    real b1 = std::pow(2, -alpha);
    real b2 = -std::exp(-2 * alpha);
    real c1 = 1, c2 = 1;


                 const  unsigned threadsPerBlock = 256;

  kernel_y1_rowsweep<<<threadsPerBlock, num_blocks(w,threadsPerBlock)>>>(w,h,alpha,imgIn,imgOut,y1,y2,a1,a2,b1,b2);
  kernel_y2_rowsweep<<<threadsPerBlock, num_blocks(w,threadsPerBlock)>>>(w,h,alpha,imgIn,imgOut,y1,y2,a3,a4,b1,b2);


    {
          dim3 block{threadsPerBlock / 32, 32, 1};
    dim3 grid{num_blocks(w, block.x), num_blocks(h, block.y), 1};
    kernel_out <<<block ,grid >>> (w,h,alpha,imgIn,imgOut,y1,y2,c1);
    }

  kernel_y1_colsweep<<<threadsPerBlock, num_blocks(h,threadsPerBlock)>>>(w,h,alpha,imgIn,imgOut,y1,y2,a5,a6,b1,b2);
  kernel_y2_colsweep<<<threadsPerBlock, num_blocks(h,threadsPerBlock)>>>(w,h,alpha,imgIn,imgOut,y1,y2,a7,a8,b1,b2);



    {
          dim3 block{threadsPerBlock / 32, 32, 1};
        dim3 grid{num_blocks(w, block.x), num_blocks(h, block.y), 1};
        kernel_out <<<block ,grid >>> (w,h,alpha,imgIn,imgOut,y1,y2,c2);
    }
}


void run(State &state, pbsize_t pbsize) {
  pbsize_t w = pbsize;                        // 4096
  pbsize_t h = pbsize / 2 + 7 * pbsize / 256; // 2160

  real alpha = 0.25;

  auto imgIn = state.allocate_array<real>({w, h}, /*fakedata*/ true, /*verify*/ false, "imgIn");
  auto imgOut = state.allocate_array<real>({w, h}, /*fakedata*/ false, /*verify*/ true, "imgOut");


     real* dev_imgIn  = state.allocate_dev<real>(w*h);
        real* dev_imgOut  = state.allocate_dev<real>(w*h);
           real* dev_y1  = state.allocate_dev<real>(w*h);
              real* dev_y2  = state.allocate_dev<real>(w*h);

  for (auto &&_ : state) {
    cudaMemcpy(dev_imgIn, imgIn.data(), w*h * sizeof(real), cudaMemcpyHostToDevice);

    kernel(w, h, alpha, dev_imgIn, dev_imgOut, dev_y1, dev_y2);

                cudaMemcpy( imgOut.data() ,dev_imgOut,  w*h* sizeof(real), cudaMemcpyDeviceToHost ); 

     cudaDeviceSynchronize();
  }

           state.free_dev(dev_imgIn);
                    state.free_dev(dev_imgOut);
                             state.free_dev(dev_y1);
                                      state.free_dev(dev_y2);
}


