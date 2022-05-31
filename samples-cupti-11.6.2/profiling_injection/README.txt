Copyright 2021 NVIDIA Corporation. All rights reserved

Profiling API injection sample code

Build this sample with

make CUDA_INSTALL_PATH=/path/to/cuda

This x86 linux-only sample contains 4 build targets:

libinjection_1.so
    * Minimal injection sample code showing how to write a dlsym wrapper
    * dlsym() is chosen as a convenient routine to wrap as it will normally be used
      by the CUDA Runtime API calls.  By wrapping it, we ensure that the injection
      code will be run.
    * dlsym() needs to still provide its original functionality; in this case, we
      use the internal glibc function __libc_dlsym and __libc_dlopen_mode to look up
      the default dlsym() function and run it, preserving original behavior

libinjection_2.so
    * Expands on the injection_1 sample to add CUPTI Callback and Profiler API calls
    * Registers callbacks for cuLaunchKernel and context creation.  This will be
      sufficient for many target applications, but others may require other launches
      to be matched, eg cuLaunchCoooperativeKernel or cuLaunchGrid.  See the Callback
      API for all possible kernel launch callbacks.
    * Creates a Profiler API configuration for each context in the target (using the
      context creation callback).  The Profiler API is configured using Kernel Replay
      and Auto Range modes with a configurable number of kernel launches within a pass.
    * The kernel launch callback is used to track how many kernels have launched in
      a given context's current pass, and if the pass reached its maximum count, it
      prints the metrics and starts a new pass.
    * At exit, any context with an unprocessed metrics (any which had partially
      completed a pass) print their data.
    * This library links in the profilerHostUtils library which may be built from the
      cuda/extras/CUPTI/samples/extensions/src/profilerhost_util/ directory

simple_target
    * Very simple executable which calls a kernel several times with increasing amount
      of work per call.

complex_target
    * More complicated example (similar to the concurrent_profiling sample) which
      launches several patterns of kernels - using default stream, multiple streams,
      and multiple devices if there are more than one device.

To use the injection library, set LD_LIBRARY_PATH and LD_PRELOAD to include that library
when you launch the target application:

env LD_PRELOAD=./libinjection_2.so LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:`pwd` ./simple_target
