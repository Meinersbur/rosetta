#include "rosetta.h"

#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <optional>
#include <string>
#include <iostream>
#include <string>
#include <charconv>
#include <fstream>
#include <filesystem>

#if ROSETTA_PLATFORM_NVIDIA
#include <cupti.h>
#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#endif

#ifdef BENCHMARK_OS_WINDOWS
#define NOMINMAX 1
#ifndef WIN32_LEAN_AND_MEAN // Already set by cupti
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <shlwapi.h>
#undef StrCat  // Don't let StrCat in string_util.h be renamed to lstrcatA
#include <versionhelpers.h>
#include <windows.h>
#else
#include <fcntl.h>
#ifndef BENCHMARK_OS_FUCHSIA
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <sys/types.h>  // this header must be included before 'sys/sysctl.h' to avoid compilation error on FreeBSD
#include <unistd.h>
#if defined BENCHMARK_OS_FREEBSD || defined BENCHMARK_OS_DRAGONFLY || \
    defined BENCHMARK_OS_MACOSX
#include <sys/sysctl.h>
#endif
#if defined(BENCHMARK_OS_MACOSX)
#include <mach/mach_init.h>
#include <mach/mach_port.h>
#include <mach/thread_act.h>
#endif
#endif

#ifdef BENCHMARK_OS_EMSCRIPTEN
#include <emscripten.h>
#endif

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <limits>
#include <mutex>


//#include "check.h"
//#include "log.h"
//#include "sleep.h"
//#include "string_util.h"
// Suppress unused warnings on helper functions.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif


namespace benchmark::internal {
int InitializeStreams() {
  static std::ios_base::Init init;
  return 0;
}
}

#if defined(BENCHMARK_OS_WINDOWS)
using usage_duration_t =  std::chrono::duration<ULONGLONG, std::ratio_multiply<std::chrono::nanoseconds::period,  std::ratio <100,1 >>> ;
#else 
using usage_duration_t =  std::chrono::microseconds;
#endif 



// TODO: filter out duplicates
// TODO: make extendable
using duration_t = std::variant<std::monostate,
    std::chrono::duration<double,std::chrono::seconds::period>,  // lowest common denominator
    std::chrono::high_resolution_clock::duration, // for wall time
    usage_duration_t , // for user/kernel time
    std::chrono::duration<double,std::chrono::milliseconds::period>, // Used by CUDA events
    std::chrono::duration<uint64_t,std::chrono::nanoseconds::period> // Used by cupti
>;





namespace {
#if defined(CLOCK_PROCESS_CPUTIME_ID) || defined(CLOCK_THREAD_CPUTIME_ID)
    static
        std::chrono::nanoseconds MakeTime(struct timespec const& ts) {
        return   std::chrono::seconds(ts.tv_sec)  + std::chrono::nanoseconds( ts.tv_nsec );
        //   return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
    }
#endif

#if defined(BENCHMARK_OS_WINDOWS)
    static
   std::pair<usage_duration_t, usage_duration_t > MakeTime(FILETIME const& kernel_time, FILETIME const& user_time) {
        ULARGE_INTEGER kernel;
        ULARGE_INTEGER user;
        kernel.HighPart = kernel_time.dwHighDateTime;
        kernel.LowPart = kernel_time.dwLowDateTime;
        user.HighPart = user_time.dwHighDateTime;
        user.LowPart = user_time.dwLowDateTime;
        return {  usage_duration_t (user.QuadPart), usage_duration_t(kernel.QuadPart )};
    }
#elif !defined(BENCHMARK_OS_FUCHSIA)
    static
        std::chrono::microseconds  MakeTime(struct timeval const& tv) {
        return std::chrono::seconds (tv.tv_sec)  +  std::chrono::microseconds (tv.tv_usec );
        //   return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
    }

    static
    std::pair<  usage_duration_t,usage_duration_t>  MakeTime(struct rusage const& ru) {
        return  { MakeTime(ru.ru_utime),MakeTime(ru.ru_stime) };
  //      return { static_cast<double>(ru.ru_utime.tv_sec) + static_cast<double>(ru.ru_utime.tv_usec) * 1e-6, static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6 };
      //  return (static_cast<double>(ru.ru_utime.tv_sec) +  static_cast<double>(ru.ru_utime.tv_usec) * 1e-6 + static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6);
    }
#endif
#if defined(BENCHMARK_OS_MACOSX)
    static
    double MakeTime(thread_basic_info_data_t const& info) {
        return (static_cast<double>(info.user_time.seconds) +
            static_cast<double>(info.user_time.microseconds) * 1e-6 +
            static_cast<double>(info.system_time.seconds) +
            static_cast<double>(info.system_time.microseconds) * 1e-6);
    }
#endif


    BENCHMARK_NORETURN static void DiagnoseAndExit(const char* msg) {
        std::cerr << "ERROR: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }

}  // end namespace

std::pair<usage_duration_t,usage_duration_t> ProcessCPUUsage() {
#if defined(BENCHMARK_OS_WINDOWS)
    HANDLE proc = GetCurrentProcess();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    if (GetProcessTimes(proc, &creation_time, &exit_time, &kernel_time, &user_time))
        return MakeTime(kernel_time, user_time);
    DiagnoseAndExit("GetProccessTimes() failed");
#elif defined(BENCHMARK_OS_EMSCRIPTEN)
    // clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ...) returns 0 on Emscripten.
    // Use Emscripten-specific API. Reported CPU time would be exactly the
    // same as total time, but this is ok because there aren't long-latency
    // synchronous system calls in Emscripten.
    return emscripten_get_now() * 1e-3;
#elif 0 // defined(CLOCK_PROCESS_CPUTIME_ID) && !defined(BENCHMARK_OS_MACOSX)
    // FIXME We want to use clock_gettime, but its not available in MacOS 10.11.
    // See https://github.com/google/benchmark/pull/292
    struct timespec spec;
    if (clock_gettime(CLOCK_PROCESS_CPUTIME_ID, &spec) == 0)
        return MakeTime(spec);
    DiagnoseAndExit("clock_gettime(CLOCK_PROCESS_CPUTIME_ID, ...) failed");
#else
    struct rusage ru;
    if (getrusage(RUSAGE_SELF, &ru) == 0) return MakeTime(ru);
    DiagnoseAndExit("getrusage(RUSAGE_SELF, ...) failed");
#endif
}


// CUDA callback_timestamp sample
#if ROSETTA_PLATFORM_NVIDIA
#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        const char* errstr;                                                    \
        cuGetErrorString(_status, &errstr);                                    \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, errstr);                     \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define CUPTI_CALL(call)                                                        \
do {                                                                            \
    CUptiResult _status = call;                                                 \
    if (_status != CUPTI_SUCCESS) {                                             \
      const char* errstr;                                                       \
      cuptiGetResultString(_status, &errstr);                                   \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",      \
              __FILE__, __LINE__, #call, errstr);                               \
      exit(-1);                                                                 \
    }                                                                           \
} while (0)

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  const char *functionName;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  size_t memcpy_bytes;
  enum cudaMemcpyKind memcpy_kind;
} RuntimeApiTrace_t;

enum launchOrder{ MEMCPY_H2D1, MEMCPY_H2D2, MEMCPY_D2H, KERNEL, THREAD_SYNC, LAUNCH_LAST};


void CUPTIAPI
getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo)
{
  static int memTransCount = 0;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t*)userdata;

switch (cbid) {
case CUPTI_RUNTIME_TRACE_CBID_cudaDriverGetVersion_v3020:
case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceProperties_v3020:
case CUPTI_RUNTIME_TRACE_CBID_cudaGetDeviceCount_v3020:
break;
default:
 fprintf(stderr, "received unknown callback: %d\n", cbid);
}


 



  // Data is collected only for the following API
  if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
      (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) ||
      (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020) ||
      (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020))  {

    // Set pointer depending on API
    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000))
    {
      traceData = traceData + KERNEL;
    }
    else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)
      traceData = traceData + THREAD_SYNC;
    else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
      traceData = traceData + MEMCPY_H2D1 + memTransCount;

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      // for a kernel launch report the kernel name, otherwise use the API
      // function name.
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)
      {
        traceData->functionName = cbInfo->symbolName;
      }
      else {
        traceData->functionName = cbInfo->functionName;
      }

      // Store parameters passed to cudaMemcpy
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
        traceData->memcpy_bytes = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->count;
        traceData->memcpy_kind = ((cudaMemcpy_v3020_params *)(cbInfo->functionParams))->kind;
      }

      // Collect timestamp for API start
      CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));

      traceData->startTimestamp = startTimestamp;
    }

    if (cbInfo->callbackSite == CUPTI_API_EXIT) {
      // Collect timestamp for API exit
      CUPTI_CALL(cuptiGetTimestamp(&endTimestamp));

      traceData->endTimestamp = endTimestamp;

      // Advance to the next memory transfer operation
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020) {
        memTransCount++;
      }
    }
  }
}

static const char *
memcpyKindStr(enum cudaMemcpyKind kind)
{
  switch (kind) {
  case cudaMemcpyHostToDevice:
    return "HostToDevice";
  case cudaMemcpyDeviceToHost:
    return "DeviceToHost";
  default:
    break;
  }

  return "<unknown>";
}

static void
displayTimestamps(RuntimeApiTrace_t *trace)
{
  // Calculate timestamp of kernel based on timestamp from
  // cudaDeviceSynchronize() call
  trace[KERNEL].endTimestamp = trace[THREAD_SYNC].endTimestamp;

  printf("startTimeStamp/Duration reported in nano-seconds\n\n");
  printf("Name\t\tStart Time\t\tDuration\tBytes\tKind\n");
  printf("%s\t%llu\t%llu\t\t%llu\t%s\n", trace[MEMCPY_H2D1].functionName,
         (unsigned long long)trace[MEMCPY_H2D1].startTimestamp,
         (unsigned long long)trace[MEMCPY_H2D1].endTimestamp - trace[MEMCPY_H2D1].startTimestamp,
         (unsigned long long)trace[MEMCPY_H2D1].memcpy_bytes,
         memcpyKindStr(trace[MEMCPY_H2D1].memcpy_kind));
  printf("%s\t%llu\t%llu\t\t%llu\t%s\n", trace[MEMCPY_H2D2].functionName,
         (unsigned long long)trace[MEMCPY_H2D2].startTimestamp,
         (unsigned long long)trace[MEMCPY_H2D2].endTimestamp - trace[MEMCPY_H2D2].startTimestamp,
         (unsigned long long)trace[MEMCPY_H2D2].memcpy_bytes,
         memcpyKindStr(trace[MEMCPY_H2D2].memcpy_kind));
  printf("%s\t%llu\t%llu\t\tNA\tNA\n", trace[KERNEL].functionName,
         (unsigned long long)trace[KERNEL].startTimestamp,
         (unsigned long long)trace[KERNEL].endTimestamp - trace[KERNEL].startTimestamp);
  printf("%s\t%llu\t%llu\t\t%llu\t%s\n", trace[MEMCPY_D2H].functionName,
         (unsigned long long)trace[MEMCPY_D2H].startTimestamp,
         (unsigned long long)trace[MEMCPY_D2H].endTimestamp - trace[MEMCPY_D2H].startTimestamp,
         (unsigned long long)trace[MEMCPY_D2H].memcpy_bytes,
         memcpyKindStr(trace[MEMCPY_D2H].memcpy_kind));
}





/*
 * Copyright 2011-2020 NVIDIA Corporation. All rights reserved
 *
 * Sample CUPTI app to print a trace of CUDA API and GPU activity
 * using asynchronous handling of activity buffers.
 *
 */


#define BUF_SIZE (32 * 1024)
#define ALIGN_SIZE (8)
#define ALIGN_BUFFER(buffer, align)                                            \
  (((uintptr_t) (buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t) (buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
    return "HtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
    return "DtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOA:
    return "HtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOH:
    return "AtoH";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOA:
    return "AtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_ATOD:
    return "AtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOA:
    return "DtoA";
  case CUPTI_ACTIVITY_MEMCPY_KIND_DTOD:
    return "DtoD";
  case CUPTI_ACTIVITY_MEMCPY_KIND_HTOH:
    return "HtoH";
  default:
    break;
  }

  return "<unknown>";
}

static const char *
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OVERHEAD_DRIVER_COMPILER:
    return "COMPILER";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_BUFFER_FLUSH:
    return "BUFFER_FLUSH";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_INSTRUMENTATION:
    return "INSTRUMENTATION";
  case CUPTI_ACTIVITY_OVERHEAD_CUPTI_RESOURCE:
    return "RESOURCE";
  default:
    break;
  }

  return "<unknown>";
}

static const char *
getActivityObjectKindString(CUpti_ActivityObjectKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return "PROCESS";
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return "THREAD";
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return "DEVICE";
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return "CONTEXT";
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return "STREAM";
  default:
    break;
  }

  return "<unknown>";
}

static uint32_t
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id)
{
  switch (kind) {
  case CUPTI_ACTIVITY_OBJECT_PROCESS:
    return id->pt.processId;
  case CUPTI_ACTIVITY_OBJECT_THREAD:
    return id->pt.threadId;
  case CUPTI_ACTIVITY_OBJECT_DEVICE:
    return id->dcs.deviceId;
  case CUPTI_ACTIVITY_OBJECT_CONTEXT:
    return id->dcs.contextId;
  case CUPTI_ACTIVITY_OBJECT_STREAM:
    return id->dcs.streamId;
  default:
    break;
  }

  return 0xffffffff;
}

static const char *
getComputeApiKindString(CUpti_ActivityComputeApiKind kind)
{
  switch (kind) {
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA:
    return "CUDA";
  case CUPTI_ACTIVITY_COMPUTE_API_CUDA_MPS:
    return "CUDA_MPS";
  default:
    break;
  }

  return "<unknown>";
}

static void
printActivity(CUpti_Activity *record)
{
  switch (record->kind)
  {
  case CUPTI_ACTIVITY_KIND_DEVICE:
    {
      CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *) record;
      printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
             "multiprocessors %u, clock %u MHz\n",
             device->name, device->id,
             device->computeCapabilityMajor, device->computeCapabilityMinor,
             (unsigned int) (device->globalMemoryBandwidth / 1024 / 1024),
             (unsigned int) (device->globalMemorySize / 1024 / 1024),
             device->numMultiprocessors, (unsigned int) (device->coreClockRate / 1000));
      break;
    }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE:
    {
      CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
      printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
             attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
      break;
    }
  case CUPTI_ACTIVITY_KIND_CONTEXT:
    {
      CUpti_ActivityContext *context = (CUpti_ActivityContext *) record;
      printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
             context->contextId, context->deviceId,
             getComputeApiKindString((CUpti_ActivityComputeApiKind) context->computeApiKind),
             (int) context->nullStreamId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMCPY:
    {
      CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *) record;
      printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
             getMemcpyKindString((CUpti_ActivityMemcpyKind) memcpy->copyKind),
             (unsigned long long) (memcpy->start - startTimestamp),
             (unsigned long long) (memcpy->end - startTimestamp),
             memcpy->deviceId, memcpy->contextId, memcpy->streamId,
             memcpy->correlationId, memcpy->runtimeCorrelationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MEMSET:
    {
      CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *) record;
      printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             memset->value,
             (unsigned long long) (memset->start - startTimestamp),
             (unsigned long long) (memset->end - startTimestamp),
             memset->deviceId, memset->contextId, memset->streamId,
             memset->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
    {
      const char* kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
      CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *) record;
      printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
             kindString,
             kernel->name,
             (unsigned long long) (kernel->start - startTimestamp),
             (unsigned long long) (kernel->end - startTimestamp),
             kernel->deviceId, kernel->contextId, kernel->streamId,
             kernel->correlationId);
      printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
             kernel->gridX, kernel->gridY, kernel->gridZ,
             kernel->blockX, kernel->blockY, kernel->blockZ,
             kernel->staticSharedMemory, kernel->dynamicSharedMemory);
      break;
    }
  case CUPTI_ACTIVITY_KIND_DRIVER:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_RUNTIME:
    {
      CUpti_ActivityAPI *api = (CUpti_ActivityAPI *) record;
      printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
             api->cbid,
             (unsigned long long) (api->start - startTimestamp),
             (unsigned long long) (api->end - startTimestamp),
             api->processId, api->threadId, api->correlationId);
      break;
    }
  case CUPTI_ACTIVITY_KIND_NAME:
    {
      CUpti_ActivityName *name = (CUpti_ActivityName *) record;
      switch (name->objectKind)
      {
      case CUPTI_ACTIVITY_OBJECT_CONTEXT:
        printf("NAME  %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      case CUPTI_ACTIVITY_OBJECT_STREAM:
        printf("NAME %s %u %s %u %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_CONTEXT),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_CONTEXT, &name->objectId),
               getActivityObjectKindString(CUPTI_ACTIVITY_OBJECT_DEVICE),
               getActivityObjectKindId(CUPTI_ACTIVITY_OBJECT_DEVICE, &name->objectId),
               name->name);
        break;
      default:
        printf("NAME %s id %u, name %s\n",
               getActivityObjectKindString(name->objectKind),
               getActivityObjectKindId(name->objectKind, &name->objectId),
               name->name);
        break;
      }
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER:
    {
      CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *) record;
      printf("MARKER id %u [ %llu ], name %s, domain %s\n",
             marker->id, (unsigned long long) marker->timestamp, marker->name, marker->domain);
      break;
    }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA:
    {
      CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *) record;
      printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
             marker->id, marker->color, marker->category,
             (unsigned long long) marker->payload.metricValueUint64,
             marker->payload.metricValueDouble);
      break;
    }
  case CUPTI_ACTIVITY_KIND_OVERHEAD:
    {
      CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *) record;
      printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
             getActivityOverheadKindString(overhead->overheadKind),
             (unsigned long long) overhead->start - startTimestamp,
             (unsigned long long) overhead->end - startTimestamp,
             getActivityObjectKindString(overhead->objectKind),
             getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
      break;
    }
  default:
    printf("  <unknown>\n");
    break;
  }
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
{
  uint8_t *bfr = (uint8_t *) malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

static
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);



void finiTrace()
{
   CUPTI_CALL(cuptiActivityFlushAll(1));
}
#endif 


using namespace std::chrono;









#ifdef _MSC_VER
//__declspec(selectany)
#else
__attribute__((weak))
#endif
void run(State& state, int n);



class IterationMeasurement {
    friend class Iteration;
    template <typename I>
    friend class Iterator;
    friend class State;
    friend class Rosetta;
    friend class Scope;
    friend class BenchmarkRun;
public:

private:
    // TODO: Make extendable (register user measures in addition to predefined ones)
    duration_t values[MeasureCount] ;
};


class BenchmarkRun {
    friend class Rosetta;
private:
#if ROSETTA_BENCH
    std::vector<IterationMeasurement> measurements;
    std::chrono::steady_clock::time_point startTime;
#endif 
#if ROSETTA_VERIFY
    int iterationsSoFar=0;
#endif 

    // TODO: Running sum, sumsquare, mean(?) of exit-determining measurement

public:


    void run(std::string program, int n) {     
#if ROSETTA_BENCH
        startTime = std::chrono::steady_clock::now();
#endif 
        State state{this};

#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
        // TODO: exclude cupti time from startTime

        // subscribe to CUPTI callbacks
        // CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &trace));

        //DRIVER_API_CALL(cuInit(0));
        //CUcontext context = 0;
        //  CUdevice device = 0;
        // DRIVER_API_CALL(cuCtxCreate(&context, 0, device));


        // CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
        // CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

        size_t attrValue = 0, attrValueSize = sizeof(size_t);
        // Device activity record is created when CUDA initializes, so we
        // want to enable it before cuInit() or any CUDA runtime call.
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
        // Enable all other activity record kinds.
        //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
        //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
        //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
        //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
        CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
        //CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

        // Register callbacks for buffer requests and for buffers completed by CUPTI.
        CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));


        uint64_t cuptiStart;
        CUPTI_CALL(cuptiGetTimestamp(&cuptiStart));
        startTimestamp = cuptiStart;
#endif 
#endif 

        // TODO: make flexible, this is just on way to find a run function
        ::run(state, n);

#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
        cuptiFinalize();
#endif
#endif 

        // TODO: print basic timing info so user knows something happened 
    }

    bool started = false;

#if ROSETTA_BENCH
    // TODO: Per-iteration data in separate object?
    std::chrono::high_resolution_clock::time_point startWall;
    usage_duration_t startUser; // in seconds; TODO: use native type
    usage_duration_t startKernel; // in seconds; TODO: use native type

#if ROSETTA_PPM_CUDA
    cudaEvent_t startCuda;
    cudaEvent_t stopCuda;
#endif

#if ROSETTA_PLATFORM_NVIDIA
    uint64_t cuptiStartHtoD, cuptiStopHtoD;
    uint64_t cuptiStartDtoH, cuptiStopDtoH;
    uint64_t cuptiStartCompute, cuptiStopCompute;
    uint64_t cuptiStartOther, cuptiStopOther;
#endif 
#endif 

    void start() {
        //printf("start\n");
        assert(!started && "Must not interleave interation scope");
        started = true;

#if ROSETTA_BENCH
#ifdef ROSETTA_PLATFORM_NVIDIA
        cuptiStartHtoD = std::numeric_limits<uint64_t>::max();
        cuptiStopHtoD = std::numeric_limits<uint64_t>::min();
        cuptiStartDtoH=  std::numeric_limits<uint64_t>::max();
        cuptiStopDtoH = std::numeric_limits<uint64_t>::min();
        cuptiStartCompute =  std::numeric_limits<uint64_t>::max();
        cuptiStopCompute = std::numeric_limits<uint64_t>::min();
        cuptiStartOther =  std::numeric_limits<uint64_t>::max();
        cuptiStopOther = std::numeric_limits<uint64_t>::min();
#endif 

#if ROSETTA_PPM_CUDA
        // TODO: Don't every time
        BENCH_CUDA_TRY(cudaEventCreate(&startCuda));
        BENCH_CUDA_TRY(cudaEventCreate(&stopCuda));
        // TODO:  flush all of L2$ as NVIDIA suggestions in synchronization.h?
#endif 

        startWall = std::chrono::high_resolution_clock::now();
        std::tie(startUser, startKernel) = ProcessCPUUsage();
#if ROSETTA_PPM_CUDA
        cudaEventRecord(startCuda);
#endif
#endif 
    }

    void stop() {
        assert(started && "No iteration active?");

#if ROSETTA_BENCH
        benchmark::ClobberMemory(); // even necessary?
       

#if ROSETTA_PPM_CUDA || ROSETTA_PLATFORM_NVIDIA
        BENCH_CUDA_TRY(cudaDeviceSynchronize());
#endif 

#if ROSETTA_PPM_CUDA
            BENCH_CUDA_TRY( cudaEventRecord(stopCuda));
#endif



        auto stopWall = std::chrono::high_resolution_clock::now();
        auto [stopUser, stopKernel] = ProcessCPUUsage();

#if ROSETTA_PPM_CUDA
        BENCH_CUDA_TRY(cudaEventRecord(stopCuda));
#endif

        auto durationWall = stopWall  - startWall;
        auto durationUser = stopUser - startUser;
        auto durationKernel = stopKernel - startKernel;
#if ROSETTA_PPM_CUDA
        float durationCuda; // ms
        BENCH_CUDA_TRY(cudaEventSynchronize(stopCuda));
        BENCH_CUDA_TRY(cudaEventElapsedTime (&durationCuda, startCuda, stopCuda));
        BENCH_CUDA_TRY(cudaEventDestroy(startCuda)); 
        BENCH_CUDA_TRY(cudaEventDestroy(stopCuda));
#else 
        float  durationCuda = 0;
#endif

        durationWall = std::max(decltype(durationWall)::zero(), durationWall);
        //durationUser = std::max(0.0, durationUser);
        //durationKernel = std::max(0.0, durationKernel);

#if ROSETTA_PLATFORM_NVIDIA
        CUPTI_CALL(cuptiActivityFlushAll(1));
#endif 
#endif 

        started =false;

#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
        auto firstEvent  =  std::min({ cuptiStartHtoD, cuptiStartDtoH, cuptiStartCompute, cuptiStartOther  }) ;
        auto lastEvent  =  std::max({cuptiStopHtoD, cuptiStopDtoH, cuptiStopCompute, cuptiStopOther  }) ;
#endif 

        IterationMeasurement m;
        m.values[WallTime] = durationWall;
        m.values[UserTime] =     std::chrono::duration<double,std::chrono::seconds::period>(durationUser);
        m.values[KernelTime] =std::chrono::duration<double,std::chrono::seconds::period>( durationKernel);
#if ROSETTA_PPM_CUDA
        m.values[AccelTime] =  std::chrono::duration<double, std::chrono::milliseconds::period>( durationCuda );
#endif
#if ROSETTA_PLATFORM_NVIDIA
        if (firstEvent <= lastEvent )
             m.values[Cupti] = std::chrono::duration<uint64_t,std::chrono::nanoseconds::period>( lastEvent -  firstEvent);
        if (cuptiStartCompute <= cuptiStopCompute )
            m.values[CuptiCompute] =std::chrono::duration<uint64_t,std::chrono::nanoseconds::period>(  cuptiStopCompute - cuptiStartCompute );
        if ( cuptiStartHtoD <= cuptiStopHtoD)
            m.values[CuptiTransferToDevice] = std::chrono::duration<uint64_t,std::chrono::nanoseconds::period>( cuptiStopHtoD - cuptiStartHtoD );
        if (  cuptiStartDtoH <=  cuptiStopDtoH)
            m.values[CuptiTransferToHost] =std::chrono::duration<uint64_t,std::chrono::nanoseconds::period>(  cuptiStopDtoH - cuptiStartDtoH);
#endif 
        measurements.push_back( std::move (m) );
#endif 
#if ROSETTA_VERIFY
        iterationsSoFar += 1;
#endif 
    }

    int refresh() {
#if ROSETTA_BENCH
        auto now = std::chrono::steady_clock::now();
        auto duration =  now - startTime;

        // TODO: configure, until stability, max/min number iterations, ...
        if (duration >= 1s) 
            return 0;
        if (measurements.size() > 10)
            return 0;

        int howManyMoreIterations = 1;
        measurements.reserve(measurements.size() + howManyMoreIterations);

        return howManyMoreIterations;
        #elif ROSETTA_VERIFY
       return 1 - iterationsSoFar;
        #else 
        #error Bench or verify it must be
        #endif 
    }

#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
     /// FIXME: Called from other threads. Require Mutex?
     void handleCuptiActivity(CUpti_Activity* record) {
         //assert(started);
         // Ignore events outside iteration
         if (!started) {
             printActivity(record);
             return;
         }

         switch (record->kind){
         case CUPTI_ACTIVITY_KIND_KERNEL:
         case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL:
         {
             CUpti_ActivityKernel5* kernel = (CUpti_ActivityKernel5*)record;
             cuptiStartCompute = std::min(cuptiStartCompute, kernel->start);
             cuptiStopCompute = std::max(cuptiStopCompute, kernel->end);
             break;
         }
         case CUPTI_ACTIVITY_KIND_MEMCPY:
         {
             // TODO: record amount of data transferred
             CUpti_ActivityMemcpy3* memcpy = (CUpti_ActivityMemcpy3*)record;
             auto copyKind = (CUpti_ActivityMemcpyKind)memcpy->copyKind;

             switch (copyKind) {
             case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
                 cuptiStartHtoD = std::min(cuptiStartHtoD, memcpy->start);
                 cuptiStopHtoD= std::max(cuptiStopHtoD, memcpy->end);
                 break;
             case CUPTI_ACTIVITY_MEMCPY_KIND_DTOH:
                 cuptiStartDtoH = std::min(cuptiStartDtoH, memcpy->start);
                 cuptiStopDtoH = std::max(cuptiStopDtoH, memcpy->end);
                 break;
             default:
                 cuptiStartOther = std::min(cuptiStartOther, memcpy->start);
                 cuptiStopOther = std::max(cuptiStopOther, memcpy->end);
                 printActivity(record);
                 break;
             }
             break;
         }

         case CUPTI_ACTIVITY_KIND_MEMSET:
         {
             CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *) record;
             cuptiStartOther = std::min(cuptiStartOther, memset->start);
             cuptiStopOther = std::max(cuptiStopOther, memset->end);
             break;
         }

         default:
             printActivity(record);
         }
    }
#endif 
#endif 
};



int State::refresh() {
    return    impl->refresh();
}

void State:: addAllocatedBytes(size_t allocatedSize) {
}


void DataHandler<double>::fake(double *data, ssize_t count) {
    // Generate some data v that is unlikely to cause problems:
    // 1. Not too close to zero (1/v)
    // 2. Not negative (sqrt(v))
    // 3. Not too large to wrap to infinity
    // 4. No special values (Inf, NaN, subnormals)
    // 5. Unlikely to cancel-out (v[0] - v[1])
    // 6. Easily representable in binary floating-point (no 0.1, 0.3)
    // TODO: don't generate same data in every call
    for (ssize_t i = 0; i < count; i+=1) 
        data[i] = 0.5 + i * (0.125);    
}


void DataHandler<double>::verify(double* data, ssize_t count) {
    for (ssize_t i = 0; i < count; i+=1) {
        // TODO: precision
        if (i > 0)
            std::cout << ' ';
        std::cout << data[i];
    }
    std::cout << '\n';
}


void Iteration::start() {
    state.impl->start();
}

void Iteration::stop() {
    // TODO: save indirection
    state.impl->stop();
}







#if ROSETTA_PLATFORM_NVIDIA
 static  CUpti_SubscriberHandle subscriber;
  static RuntimeApiTrace_t trace[LAUNCH_LAST];
#endif 


  static const  char *getUnit(Measure measure) {
      switch (measure) {
      case WallTime:
      case UserTime:
      case KernelTime:
          return "s";
      case OpenMPWTime:
          return "s";
      case AccelTime:
          return "ms";
      case Cupti:
      case CuptiCompute:
      case CuptiTransferToDevice:
      case CuptiTransferToHost:
          return "ns";
      }
      assert(!"What's the time unit?");
      return nullptr;
  }


  template <typename... LAMBDAS> struct visitors : LAMBDAS... {
      using LAMBDAS::operator()...;
  };

  template <typename... LAMBDAS> visitors(LAMBDAS... x) -> visitors<LAMBDAS...>;
  template <class... T>
  constexpr bool always_false = false;

  struct duration_formatter {
      std::ostringstream& buf;

      template <typename T>
      typename std::enable_if<std::is_floating_point <T>::value>::type  printNumber(T x) {
          buf << std::setprecision(std::numeric_limits<double>::digits10 + 1) << x;
      }

      template <typename T>
      typename std::enable_if<std::is_integral <T>::value>::type printNumber(T x) {
          buf << x;
      }

      template<typename Ratio>
      void printUnit() {
      }

      template <typename T,  typename Ratio>
      void operator() (std::chrono::duration<T,Ratio> v) {
          printNumber(v.count());

          if constexpr (std::is_same_v<Ratio, std::chrono::seconds::period >) {
              buf << " s";
          }
          else if   constexpr (std::is_same_v<Ratio, std::chrono::nanoseconds::period >) {
              buf << " ns";
          } else if   constexpr (std::is_same_v<Ratio, std::chrono::milliseconds::period >) {
              buf << " ms";
          } else {
              static_assert(always_false<T>, "Unhandled time unit");
          }
      }

      template <typename T>
      void operator() (std::chrono::duration<T, usage_duration_t::period> v) {
          // fallback to nanoseconds
          return this->operator()( std::chrono::duration_cast<std::chrono::nanoseconds>(v)   );
      }


      void operator() (std::monostate) {  }
  } ;



  static std::string formatDuration(duration_t duration) {
      std::ostringstream buf;
      if (std::holds_alternative<std::monostate>(duration))
          return "";       // Should not have been called

      duration_formatter callme{buf};

      std::visit( callme, duration);



      return buf.str();
  }

// TODO: make singleton?
struct Rosetta {
    static constexpr const char* measureDesc[MeasureCount] = {"Wall Clock", "User", "Kernel", "GPU"};
    static constexpr const char* measureName[MeasureCount] = {"walltime", "usertime", "kerneltime", "openmp","acceltime", "cupti", "cupti_compute", "cupti_todev", "cupti_fromdev" };

    static std::string  escape(std::string s) {
        return s; // TODO
    }

   static  BenchmarkRun *currentRun ;
  static void run(std::string program, int n) {       
            BenchmarkRun executor;
            currentRun = &executor;
            executor.run(program, n);

#if ROSETTA_BENCH
            int numMeasures = executor.measurements.size();
            int startMeasures = 0;
            if (numMeasures >=2) {
                // Remove cold run if we can afford it.
                startMeasures = 1;
            }


                std::cout << R"(<?xml version="1.0" encoding="UTF-8" ?>)" << std::endl;
                std::cout << R"(<benchmarks>)" <<std::endl;
                std::cout << R"(  <benchmark name=")" << escape(program) <<   R"(" n=")" << n << "\" cold_iterations=\"" << startMeasures <<  R"(">)"<<std::endl;
                for (int i = startMeasures; i < numMeasures; i+=1) {
                    auto &m = executor. measurements[i];
               // for (auto &m :executor. measurements) {
                    // TODO: custom times and units
                    std::cout << "    <iteration";

                    for (int i = 0; i <= MeasureLast; i+=1) {
                        auto measure = (Measure)i;
                        auto &val = m.values[measure];
                        if (std::holds_alternative<std::monostate>(val)) continue;
                        std::cout <<' ' << measureName[measure] << "=\"" << formatDuration(m.values[measure]) << '\"';
                    }


                    std::cout << R"(/>)"<<std::endl;  

                }
                // TODO: run properties: num threads, device, executable hash, allocated bytes, num flop (calculated), num updates, performance counters, ..
                std::cout << R"(  </benchmark>)" <<std::endl;
                std::cout << R"(</benchmarks>)" <<std::endl;
#endif 

                currentRun = nullptr;
        }

#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
        static void handleCuptiActivity(CUpti_Activity *record) {
            if (!currentRun) {
                // Activity not asociated with a run
                return ;
            }


            currentRun->handleCuptiActivity(record);
        }
#endif 
#endif 
};


BenchmarkRun *Rosetta::currentRun =nullptr;



#if ROSETTA_BENCH
#if ROSETTA_PLATFORM_NVIDIA
static
void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
    CUptiResult status;
    CUpti_Activity *record = NULL;

    if (validSize > 0) {
        do {
            status = cuptiActivityGetNextRecord(buffer, validSize, &record);
            if (status == CUPTI_SUCCESS) {
                Rosetta::handleCuptiActivity(record);
            }
            else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
                break;
            else {
                CUPTI_CALL(status);
            }
        } while (1);

        // report any records dropped from the queue
        size_t dropped;
        CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
        if (dropped != 0) {
            printf("Dropped %u activity records\n", (unsigned int) dropped);
        }
    }

    free(buffer);
}
#endif 
#endif 


std::tuple <std::string_view, std::optional<  std::string_view>> nextArg(int argc, char* argv[], int &i) {
    std::string_view arg{argv[i]};
    i +=1 ;
    if ( arg.length() > 3 && arg.substr(0, 2) == "--" && isalpha( arg[2])) {
        std::string_view sw = arg.substr(2);
        auto eqpos = sw.find('=');
        if (eqpos != std::string_view::npos) {
            auto swname = sw.substr(0, eqpos);
            auto swarg = sw.substr(eqpos + 1);
            return {swname, swarg };
        }
        return { sw, {} };
    }
    if ( arg.length() > 2 && arg[0] == '-' && isalpha( arg[1]) ) {    
        auto swname =    arg.substr(1,1);
        if (arg.length() > 2) {      
            auto swarg =arg.substr(2);
            return {swname, swarg };
        }
        return { swname, {} };
    }

    // Not recognized as a switch. Assume positional argument.
    return { std::string_view(), arg };
}


static  std::string_view trim(std::string_view str){ constexpr char const* whitespace { " \t\r\n" };
    auto start =  str.find_first_not_of(whitespace);
    if (start == std::string::npos) return ""; // contains only whitespace chars
    auto stop = str.find_last_not_of(whitespace,   str.length());
    if (stop == std::string::npos) return ""; // contains only whitespace chars
    if (stop < start ) return "";
    return  str.substr(start, stop - start + 1); 
}


static 
int parseInt(std::string_view s) {
    int result;
    auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.length(), result );
    // TODO: check for error
    return result;
}

int main(int argc, char* argv[]) {
    assert(argc >= 1);

    // TODO: Get benchname from program itself.
    std::filesystem ::path program ( argv[0]);
    auto progname = program.filename().string();

   auto dotpos = progname.find_first_of('.');
   auto benchname = progname;
   if (dotpos != std::string::npos) {
    benchname = progname.substr(0, dotpos);
   }

    std::string_view problemsizefile ;
    int problemsize = -1;
    int i =1;
    while (i < argc) {
        auto [name,val] = nextArg(argc, argv, i);

        if (name == "n") {
            if (!val.has_value() && i < argc) {
                val = argv[i]; i+= 1;
            }
            problemsize = parseInt(*val);
        }
        else if (name == "problemsizefile") {
            if (!val.has_value() && i < argc) {
                val = argv[i]; i+= 1;
            }
            problemsizefile = *val;
        }
    }

   
    // TODO: default problem size
    if (problemsize >= 0) {
        // Explicit problemsize has priority
    } else if (!problemsizefile.empty()) {
        std::vector<std::string> lines;
        {
            std::ifstream psfile(std::string(problemsizefile).c_str(), std::ios::in);
            std::string myline;
            for (std::string myline; std::getline(psfile, myline); )
                lines.push_back(myline);
        }

        std::vector<std::string> seclines;
        int i =0 ;
        auto secheader  = "[" + benchname + "]";
        while (i < lines.size()) {
            if (trim(lines[i]) == secheader) {
                i += 1;
                while (i < lines.size()) {
                    auto trimmed = trim(lines[i]);
                    if (trimmed.substr(0, 1) == "[") break;
                    if (!trimmed.empty())
                        seclines.push_back( std::string (trimmed)); 
                    i+=1;
                }
                break;
             }
            i+=1;
        }


        for (auto &&line : seclines) {
            auto lineview = std::string_view(line);
            auto eqpos = line.find_first_of('=');
            if (eqpos == std::string::npos) continue;
            auto key = trim(  lineview.substr(0, eqpos) );
            auto val = trim( lineview.substr(eqpos + 1) );

            if (key == "n") {
                auto nval = parseInt(val);
                problemsize = nval;
                continue;
            }
        }
    }

#if 0
    for (int i = 1; i < argc; ++i) {
        std::string_view a{argv[i]};
        if (a == "-n") {
        }else if (a.st)
    }


   // std::cerr <<program << "\n";
    argc-=1;
    argv += 1;

    // TODO: benchmark-specific default size
    int n = 100;
    if (argc > 0) {
       // std::cerr <<argv[0] << "\n";
       n = std::atoi(argv[0]);
       argc -= 1;
       argv += 1;
    }
#endif 


    assert(problemsize >= 1);
    Rosetta::run( program.filename().string(), problemsize );

    return EXIT_SUCCESS;
}

