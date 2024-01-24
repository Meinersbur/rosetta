// BUILD: add_benchmark(cuda,
// BUILD:                SizeParam('n', pbsize, runtime, verify=129, train=1024, ref=6000, min=0),
// BUILD:                TuneParam('THREADS_PER_BLOCK', int, runtime, choices=[32,64,128,256,512])
// BUILD:              )

#include <rosetta.h>


__global__ void histo_prescan_kernel (unsigned int* input, int size, unsigned int* minmax)
{
    __shared__ float Avg[PRESCAN_THREADS];
    __shared__ float StdDev[PRESCAN_THREADS];

    int stride = size/gridDim.x;
    int addr = blockIdx.x*stride+threadIdx.x;
    int end = blockIdx.x*stride + stride/8; // Only sample 1/8th of the input data

    // Compute the average per thread
    float avg = 0.0;
    unsigned int count = 0;
    while (addr < end){
        avg += input[addr];
        count++;
	addr += blockDim.x;
    }
    avg /= count;
    Avg[threadIdx.x] = avg;

    // Compute the standard deviation per thread
    int addr2 = blockIdx.x*stride+threadIdx.x;
    float stddev = 0;
    while (addr2 < end){
        stddev += (input[addr2]-avg)*(input[addr2]-avg);
        addr2 += blockDim.x;
    }
    stddev /= count;
    StdDev[threadIdx.x] = sqrtf(stddev);

#define SUM(stride__)\
if(threadIdx.x < stride__){\
    Avg[threadIdx.x] += Avg[threadIdx.x+stride__];\
    StdDev[threadIdx.x] += StdDev[threadIdx.x+stride__];\
}

    // Add all the averages and standard deviations from all the threads
    // and take their arithmetic average (as a simplified approximation of the
    // real average and standard deviation.
#if (PRESCAN_THREADS >= 32)    
    for (int stride = PRESCAN_THREADS/2; stride >= 32; stride = stride >> 1){
	__syncthreads();
	SUM(stride);
    }
#endif
#if (PRESCAN_THREADS >= 16)
    SUM(16);
#endif
#if (PRESCAN_THREADS >= 8)
    SUM(8);
#endif
#if (PRESCAN_THREADS >= 4)
    SUM(4);
#endif
#if (PRESCAN_THREADS >= 2)
    SUM(2);
#endif

    if (threadIdx.x == 0){
        float avg = Avg[0]+Avg[1];
	avg /= PRESCAN_THREADS;
	float stddev = StdDev[0]+StdDev[1];
	stddev /= PRESCAN_THREADS;

        // Take the maximum and minimum range from all the blocks. This will
        // be the final answer. The standard deviation is taken out to 10 sigma
        // away from the average. The value 10 was obtained empirically.
	    atomicMin(minmax,((unsigned int)(avg-10*stddev))/(KB*1024));
        atomicMax(minmax+1,((unsigned int)(avg+10*stddev))/(KB*1024));
    }
}


__device__ void calculateBin (
        const unsigned int bin,
        uchar4 *sm_mapping)
{
        unsigned char offset  =  bin        %   4;
        unsigned char indexlo = (bin >>  2) % 256;
        unsigned char indexhi = (bin >> 10) %  KB;
        unsigned char block   =  bin / BINS_PER_BLOCK;

        offset *= 8;

        uchar4 sm;
        sm.x = block;
        sm.y = indexhi;
        sm.z = indexlo;
        sm.w = offset;

        *sm_mapping = sm;
}

__global__ void histo_intermediates_kernel (
        uint2 *input,
        unsigned int height,
        unsigned int width,
        unsigned int input_pitch,
        uchar4 *sm_mappings)
{
        unsigned int line = UNROLL * blockIdx.x;// 16 is the unroll factor;

        uint2 *load_bin = input + line * input_pitch + threadIdx.x;

        unsigned int store = line * width + threadIdx.x;
        bool skip = (width % 2) && (threadIdx.x == (blockDim.x - 1));

        #pragma unroll
        for (int i = 0; i < UNROLL; i++)
        {
                uint2 bin_value = *load_bin;

                calculateBin (
                        bin_value.x,
                        &sm_mappings[store]
                );

                if (!skip) calculateBin (
                        bin_value.y,
                        &sm_mappings[store + blockDim.x]
                );

                load_bin += input_pitch;
                store += width;
        }
}



__device__ void testIncrementGlobal (
        unsigned int *global_histo,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        const uchar4 sm)
{
        const unsigned int range = sm.x;
        const unsigned int indexhi = sm.y;
        const unsigned int indexlo = sm.z;
        const unsigned int offset  = sm.w;

        /* Scan for inputs that are outside the central region of histogram */
        if (range < sm_range_min || range > sm_range_max)
        {
                const unsigned int bin = range * BINS_PER_BLOCK + offset / 8 + (indexlo << 2) + (indexhi << 10);
                const unsigned int bin_div2 = bin / 2;
                const unsigned int bin_offset = (bin % 2 == 1) ? 16 : 0;

                unsigned int old_val = global_histo[bin_div2];
                unsigned short old_bin = (old_val >> bin_offset) & 0xFFFF;

                if (old_bin < 255)
                {
                        atomicAdd (&global_histo[bin_div2], 1 << bin_offset);
                }
        }
}

__device__ void testIncrementLocal (
        unsigned int *global_overflow,
        unsigned int smem[KB][256],
        const unsigned int myRange,
        const uchar4 sm)
{
        const unsigned int range = sm.x;
        const unsigned int indexhi = sm.y;
        const unsigned int indexlo = sm.z;
        const unsigned int offset  = sm.w;

        /* Scan for inputs that are inside the central region of histogram */
        if (range == myRange)
        {
                /* Atomically increment shared memory */
                unsigned int add = (unsigned int)(1 << offset);
                unsigned int prev = atomicAdd (&smem[indexhi][indexlo], add);

                /* Check if current bin overflowed */
                unsigned int prev_bin_val = (prev >> offset) & 0x000000FF;

                /* If there was an overflow, record it and record if it cascaded into other bins */
                if (prev_bin_val == 0x000000FF)
                {
                        const unsigned int bin =
                                range * BINS_PER_BLOCK +
                                offset / 8 + (indexlo << 2) + (indexhi << 10);

                        bool can_overflow_to_bin_plus_1 = (offset < 24) ? true : false;
                        bool can_overflow_to_bin_plus_2 = (offset < 16) ? true : false;
                        bool can_overflow_to_bin_plus_3 = (offset <  8) ? true : false;

                        bool overflow_into_bin_plus_1 = false;
                        bool overflow_into_bin_plus_2 = false;
                        bool overflow_into_bin_plus_3 = false;

                        unsigned int prev_bin_plus_1_val = (prev >> (offset +  8)) & 0x000000FF;
                        unsigned int prev_bin_plus_2_val = (prev >> (offset + 16)) & 0x000000FF;
                        unsigned int prev_bin_plus_3_val = (prev >> (offset + 24)) & 0x000000FF;

                        if (can_overflow_to_bin_plus_1 &&        prev_bin_val == 0x000000FF) overflow_into_bin_plus_1 = true;
                        if (can_overflow_to_bin_plus_2 && prev_bin_plus_1_val == 0x000000FF) overflow_into_bin_plus_2 = true;
                        if (can_overflow_to_bin_plus_3 && prev_bin_plus_2_val == 0x000000FF) overflow_into_bin_plus_3 = true;

                        unsigned int bin_plus_1_add;
                        unsigned int bin_plus_2_add;
                        unsigned int bin_plus_3_add;

                        if (overflow_into_bin_plus_1) bin_plus_1_add = (prev_bin_plus_1_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_2) bin_plus_2_add = (prev_bin_plus_2_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;
                        if (overflow_into_bin_plus_3) bin_plus_3_add = (prev_bin_plus_3_val < 0x000000FF) ? 0xFFFFFFFF : 0x000000FF;

                                                      atomicAdd (&global_overflow[bin],   256);
                        if (overflow_into_bin_plus_1) atomicAdd (&global_overflow[bin+1], bin_plus_1_add);
                        if (overflow_into_bin_plus_2) atomicAdd (&global_overflow[bin+2], bin_plus_2_add);
                        if (overflow_into_bin_plus_3) atomicAdd (&global_overflow[bin+3], bin_plus_3_add);
                }
        }
}

__device__ void clearMemory (unsigned int smem[KB][256])
{
        for (int i = threadIdx.x; i < BINS_PER_BLOCK / 4; i += blockDim.x)
        {
                ((unsigned int*)smem)[i] = 0;
        }
}

__device__ void copyMemory (unsigned int *dst, unsigned int src[KB][256])
{
        for (int i = threadIdx.x; i < BINS_PER_BLOCK / 4; i += blockDim.x)
        {
                dst[i] = ((unsigned int*)src)[i];
        }
}

__global__ void histo_main_kernel (
        uchar4 *sm_mappings,
        unsigned int num_elements,
        unsigned int sm_range_min,
        unsigned int sm_range_max,
        unsigned int histo_height,
        unsigned int histo_width,
        unsigned int *global_subhisto,
        unsigned int *global_histo,
        unsigned int *global_overflow)
{
        /* Most optimal solution uses 24 * 1024 bins per threadblock */
        __shared__ unsigned int sub_histo[KB][256];

        /* Each threadblock contributes to a specific 24KB range of histogram,
         * and also scans every N-th line for interesting data.  N = gridDim.x
         */
        unsigned int local_scan_range = sm_range_min + blockIdx.y;
        unsigned int local_scan_load = blockIdx.x * blockDim.x + threadIdx.x;

        clearMemory (sub_histo);
        __syncthreads();

        if (blockIdx.y == 0)
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        /* Read buffer */
                        uchar4 sm = sm_mappings[local_scan_load];
                        local_scan_load += blockDim.x * gridDim.x;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                sub_histo,
                                local_scan_range,
                                sm
                        );
                        testIncrementGlobal (
                                global_histo,
                                sm_range_min,
                                sm_range_max,
                                sm
                        );
                }
        }
        else
        {
                /* Loop through and scan the input */
                while (local_scan_load < num_elements)
                {
                        /* Read buffer */
                        uchar4 sm = sm_mappings[local_scan_load];
                        local_scan_load += blockDim.x * gridDim.x;

                        /* Check input */
                        testIncrementLocal (
                                global_overflow,
                                sub_histo,
                                local_scan_range,
                                sm
                        );
                }
        }

        /* Store sub histogram to global memory */
        unsigned int store_index = blockIdx.x * (histo_height * histo_width / 4) + (local_scan_range * BINS_PER_BLOCK / 4);

        __syncthreads();
        copyMemory (&(global_subhisto[store_index]), sub_histo);
}


/* Combine all the sub-histogram results into one final histogram */
__global__ void histo_final_kernel (
    unsigned int sm_range_min, 
    unsigned int sm_range_max,
    unsigned int histo_height, 
    unsigned int histo_width,
    unsigned int *global_subhisto,
    unsigned int *global_histo,
    unsigned int *global_overflow,
    unsigned int *final_histo) //final output
{
    unsigned int start_offset = threadIdx.x + blockIdx.x * blockDim.x;
    const ushort4 zero_short  = {0, 0, 0, 0};
    const uint4 zero_int      = {0, 0, 0, 0};

    unsigned int size_low_histo = sm_range_min * BINS_PER_BLOCK;
    unsigned int size_mid_histo = (sm_range_max - sm_range_min +1) * BINS_PER_BLOCK;

    /* Clear lower region of global histogram */
    for (unsigned int i = start_offset; i < size_low_histo/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        ((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the middle region of the overflow buffer */
    for (unsigned int i = (size_low_histo/4) + start_offset; i < (size_low_histo+size_mid_histo)/4; i += gridDim.x * blockDim.x)
    {
        uint4 global_histo_data = ((uint4*)global_overflow)[i];
        ((uint4*)global_overflow)[i] = zero_int;

        uint4 internal_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        #pragma unroll
        for (int j = 0; j < BLOCK_X; j++)
        {
            unsigned int bin4in = ((unsigned int*)global_subhisto)[i + j * histo_height * histo_width / 4];
            internal_histo_data.x += (bin4in >>  0) & 0xFF;
            internal_histo_data.y += (bin4in >>  8) & 0xFF;
            internal_histo_data.z += (bin4in >> 16) & 0xFF;
            internal_histo_data.w += (bin4in >> 24) & 0xFF;
        }

        internal_histo_data.x = min (internal_histo_data.x, 255);
        internal_histo_data.y = min (internal_histo_data.y, 255);
        internal_histo_data.z = min (internal_histo_data.z, 255);
        internal_histo_data.w = min (internal_histo_data.w, 255);

        uchar4 final_histo_data = {
            internal_histo_data.x,
            internal_histo_data.y,
            internal_histo_data.z,
            internal_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }

    /* Clear the upper region of global histogram */
    for (unsigned int i = ((size_low_histo+size_mid_histo)/4) + start_offset; i < (histo_height*histo_width)/4; i += gridDim.x * blockDim.x)
    {
        ushort4 global_histo_data = ((ushort4*)global_histo)[i];
        ((ushort4*)global_histo)[i] = zero_short;

        global_histo_data.x = min (global_histo_data.x, 255);
        global_histo_data.y = min (global_histo_data.y, 255);
        global_histo_data.z = min (global_histo_data.z, 255);
        global_histo_data.w = min (global_histo_data.w, 255);

        uchar4 final_histo_data = {
            global_histo_data.x,
            global_histo_data.y,
            global_histo_data.z,
            global_histo_data.w
        };

        ((uchar4*)final_histo)[i] = final_histo_data;
    }
}

void run(State &state, pbsize_t n) {
  pbsize_t img_width = n;
  pbsize_t img_height = n - n/3;
  pbsize_t hist_width = 255;
  pbsize_t hist_height = 3;

  auto img = state.allocate_array<unsigned>({img_width*img_height}, /*fakedata*/ true, /*verify*/ false, "img");
  auto histo = state.allocate_array<uint8_t>({hist_width*hist_height}, /*fakedata*/ false, /*verify*/ true, "histo");
  
  unsigned *dev_img = state.allocate_dev<unsigned>(img_width*img_height);
  uint8_t *dev_histo = state.allocate_dev<uint8_t>(hist_width*hist_height);

  for (auto &&_ : state) {
    unsigned int ranges_h[2] = {UINT32_MAX, 0};
    BENCH_CUDA_TRY(cudaMemcpy(dev_img, img.data(), img_width*img_height * sizeof(unsigned), cudaMemcpyHostToDevice));

    cudaMemcpy(ranges,ranges_h, 2*sizeof(unsigned int), cudaMemcpyHostToDevice);
    
    histo_prescan_kernel<<<dim3(PRESCAN_BLOCKS_X),dim3(PRESCAN_THREADS)>>>((unsigned int*)input, img_height*img_width, ranges);

    cudaMemcpy(ranges_h,ranges, 2*sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemset(global_subhisto,0,img_width*histo_height*sizeof(unsigned int));
    
    histo_intermediates_kernel<<<dim3((img_height + UNROLL-1)/UNROLL), dim3((img_width+1)/2)>>>(
                (uint2*)(input),
                (unsigned int)img_height,
                (unsigned int)img_width,
                (img_width+1)/2,
                (uchar4*)(sm_mappings)
    );
    
    histo_main_kernel<<<dim3(BLOCK_X, ranges_h[1]-ranges_h[0]+1), dim3(THREADS)>>>(
                (uchar4*)(sm_mappings),
                img_height*img_width,
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow)    
    );
    
    histo_final_kernel<<<dim3(BLOCK_X*3), dim3(512)>>>(
                ranges_h[0], ranges_h[1],
                histo_height, histo_width,
                (unsigned int*)(global_subhisto),
                (unsigned int*)(global_histo),
                (unsigned int*)(global_overflow),
                (unsigned int*)(final_histo)
    );

    BENCH_CUDA_TRY(cudaMemcpy(histo.data(), dev_histo, n * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    BENCH_CUDA_TRY(cudaDeviceSynchronize());
  }

  state.free_dev(dev_data);
  state.free_dev(dev_histo);
}
