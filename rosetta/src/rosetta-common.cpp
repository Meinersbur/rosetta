#include "rosetta.h"

#include "rosetta-common.h"
#include "rosetta-stat.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>

//#if HAS_INCLUDE_CHARCONV
//#include <charconv>
//#endif

#if ROSETTA_PPM_NVIDIA
#include <cuda.h>
#endif

#if ROSETTA_PLATFORM_NVIDIA
#include <cupti.h>
#include <stdio.h>
#endif

#if ROSETTA_PPM_OPENMP || ROSETTA_PLATFORM_OPENMP
#include <omp.h>
#endif

#if ROSETTA_PPM_SYCL
#include <sycl/sycl.hpp>
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


using namespace rosetta;

namespace benchmark::internal {
int InitializeStreams() {
  static std::ios_base::Init init;
  return 0;
}
} // namespace benchmark::internal


static double to_seconds(const duration_t &lhs) {
  return std::visit(
      [](const auto &lhs) {
        using T = std::remove_const_t<std::remove_reference_t<decltype(lhs)>>;
        if constexpr (!std::is_same_v<T, std::monostate>)
          return std::chrono::duration_cast<std::chrono::duration<double>>(lhs).count();
        return 0.0;
      },
      lhs);
}



// size in bytes
using memory_t = ssize_t;



namespace {
#if defined(CLOCK_PROCESS_CPUTIME_ID) || defined(CLOCK_THREAD_CPUTIME_ID)
static std::chrono::nanoseconds MakeTime(struct timespec const &ts) {
  return std::chrono::seconds(ts.tv_sec) + std::chrono::nanoseconds(ts.tv_nsec);
  //   return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
}
#endif

#if defined(BENCHMARK_OS_WINDOWS)
static std::pair<usage_duration_t, usage_duration_t> MakeTime(FILETIME const &kernel_time, FILETIME const &user_time) {
  ULARGE_INTEGER kernel;
  ULARGE_INTEGER user;
  kernel.HighPart = kernel_time.dwHighDateTime;
  kernel.LowPart = kernel_time.dwLowDateTime;
  user.HighPart = user_time.dwHighDateTime;
  user.LowPart = user_time.dwLowDateTime;
  return {usage_duration_t(user.QuadPart), usage_duration_t(kernel.QuadPart)};
}
#elif !defined(BENCHMARK_OS_FUCHSIA)
static std::chrono::microseconds MakeTime(struct timeval const &tv) {
  return std::chrono::seconds(tv.tv_sec) + std::chrono::microseconds(tv.tv_usec);
  //   return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
}

static std::pair<usage_duration_t, usage_duration_t> MakeTime(struct rusage const &ru) {
  return {MakeTime(ru.ru_utime), MakeTime(ru.ru_stime)};
  //      return { static_cast<double>(ru.ru_utime.tv_sec) + static_cast<double>(ru.ru_utime.tv_usec) * 1e-6, static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6 };
  //  return (static_cast<double>(ru.ru_utime.tv_sec) +  static_cast<double>(ru.ru_utime.tv_usec) * 1e-6 + static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6);
}
#endif
#if defined(BENCHMARK_OS_MACOSX)
static double MakeTime(thread_basic_info_data_t const &info) {
  return (static_cast<double>(info.user_time.seconds) +
          static_cast<double>(info.user_time.microseconds) * 1e-6 +
          static_cast<double>(info.system_time.seconds) +
          static_cast<double>(info.system_time.microseconds) * 1e-6);
}
#endif


BENCHMARK_NORETURN static void DiagnoseAndExit(const char *msg) {
  std::cerr << "ERROR: " << msg << std::endl;
  std::exit(EXIT_FAILURE);
}



} // end namespace

std::pair<usage_duration_t, usage_duration_t> ProcessCPUUsage() {
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
  if (getrusage(RUSAGE_SELF, &ru) == 0)
    return MakeTime(ru);
  DiagnoseAndExit("getrusage(RUSAGE_SELF, ...) failed");
#endif
}


memory_t getMaxRSS() {
#if defined(BENCHMARK_OS_WINDOWS)

  // https://docs.microsoft.com/en-us/windows/win32/api/psapi/ns-psapi-process_memory_counters
  PROCESS_MEMORY_COUNTERS meminfo = {0};
  GetProcessMemoryInfo(GetCurrentProcess(), &meminfo, sizeof(meminfo));
  return meminfo.PeakWorkingSetSize;

#if 0
    // https://github.com/giampaolo/psutil/blob/master/psutil/_psutil_windows.c

    NTSTATUS status;
    PVOID buffer;
    SIZE_T bufferSize;

    bufferSize = 0x8000;
    buffer = malloc(bufferSize);
    assert(buffer);

    auto hProcess =NtCurrentProcess ();
    while ((status = NtQueryVirtualMemory(
        hProcess,
        NULL,
        MemoryWorkingSetInformation,
        buffer,
        bufferSize,
        NULL)) == STATUS_INFO_LENGTH_MISMATCH)
    {
        bufferSize *= 2;
        // Fail if we're resizing the buffer to something very large.
        if (bufferSize > 256 * 1024 * 1024) {
            assert(!                "NtQueryVirtualMemory bufsize is too large");
            return 1;
        }
        buffer = realloc(buffer, bufferSize);
            assert(buffer);
    }

    assert(NT_SUCCESS(status));
   

    auto wSetInfo = (PMEMORY_WORKING_SET_INFORMATION)buffer;


    memory_t pages = 0 ;
    for (int i = 0; i < wSetInfo->NumberOfEntries; i++) {
        // This is what ProcessHacker does.
        /*
        wsCounters.NumberOfPages++;
        if (wsInfo->WorkingSetInfo[i].ShareCount > 1)
        wsCounters.NumberOfSharedPages++;
        if (wsInfo->WorkingSetInfo[i].ShareCount == 0)
        wsCounters.NumberOfPrivatePages++;
        if (wsInfo->WorkingSetInfo[i].Shared)
        wsCounters.NumberOfShareablePages++;
        */

        // This is what psutil does: count shared pages that only one process is using as private (USS).
        /*
        if (!wsInfo->WorkingSetInfo[i].Shared ||
            wsInfo->WorkingSetInfo[i].ShareCount <= 1) {
            wsCounters.NumberOfPrivatePages++;
        }*/

        auto &wsInfo =wSetInfo->WorkingSetInfo[i];
        if (wsInfo.Shared ||  wsInfo.ShareCount <= 1)
            pages += 1;
    }

    HeapFree(GetProcessHeap(), 0, wSetInfo);
    free(buffer);
    return 0;
#endif

#else
  struct rusage ru = {0}; // TODO: Just one call with ProcessCPUUsage
  getrusage(RUSAGE_SELF, &ru);
  return ru.ru_maxrss * 1024;
#endif
}


// CUDA callback_timestamp sample
#if ROSETTA_PLATFORM_NVIDIA
#define DRIVER_API_CALL(apiFuncCall)                                       \
  do {                                                                     \
    CUresult _status = apiFuncCall;                                        \
    if (_status != CUDA_SUCCESS) {                                         \
      const char *errstr;                                                  \
      cuGetErrorString(_status, &errstr);                                  \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #apiFuncCall, errstr);                   \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                         \
  do {                                                                        \
    cudaError_t _status = apiFuncCall;                                        \
    if (_status != cudaSuccess) {                                             \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",    \
              __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status)); \
      exit(-1);                                                               \
    }                                                                         \
  } while (0)

#define CUPTI_CALL(call)                                                   \
  do {                                                                     \
    CUptiResult _status = call;                                            \
    if (_status != CUPTI_SUCCESS) {                                        \
      const char *errstr;                                                  \
      cuptiGetResultString(_status, &errstr);                              \
      fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n", \
              __FILE__, __LINE__, #call, errstr);                          \
      exit(-1);                                                            \
    }                                                                      \
  } while (0)

// Structure to hold data collected by callback
typedef struct RuntimeApiTrace_st {
  const char *functionName;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  size_t memcpy_bytes;
  enum cudaMemcpyKind memcpy_kind;
} RuntimeApiTrace_t;

enum launchOrder { MEMCPY_H2D1,
                   MEMCPY_H2D2,
                   MEMCPY_D2H,
                   KERNEL,
                   THREAD_SYNC,
                   LAUNCH_LAST };


void CUPTIAPI
getTimestampCallback(void *userdata, CUpti_CallbackDomain domain,
                     CUpti_CallbackId cbid, const CUpti_CallbackData *cbInfo) {
  static int memTransCount = 0;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  RuntimeApiTrace_t *traceData = (RuntimeApiTrace_t *)userdata;

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
      (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)) {

    // Set pointer depending on API
    if ((cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020) ||
        (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000)) {
      traceData = traceData + KERNEL;
    } else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaDeviceSynchronize_v3020)
      traceData = traceData + THREAD_SYNC;
    else if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy_v3020)
      traceData = traceData + MEMCPY_H2D1 + memTransCount;

    if (cbInfo->callbackSite == CUPTI_API_ENTER) {
      // for a kernel launch report the kernel name, otherwise use the API
      // function name.
      if (cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunch_v3020 ||
          cbid == CUPTI_RUNTIME_TRACE_CBID_cudaLaunchKernel_v7000) {
        traceData->functionName = cbInfo->symbolName;
      } else {
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
memcpyKindStr(enum cudaMemcpyKind kind) {
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
displayTimestamps(RuntimeApiTrace_t *trace) {
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
#define ALIGN_BUFFER(buffer, align) \
  (((uintptr_t)(buffer) & ((align)-1)) ? ((buffer) + (align) - ((uintptr_t)(buffer) & ((align)-1))) : (buffer))

// Timestamp at trace initialization time. Used to normalized other
// timestamps
static uint64_t startTimestamp;

static const char *
getMemcpyKindString(CUpti_ActivityMemcpyKind kind) {
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
getActivityOverheadKindString(CUpti_ActivityOverheadKind kind) {
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
getActivityObjectKindString(CUpti_ActivityObjectKind kind) {
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
getActivityObjectKindId(CUpti_ActivityObjectKind kind, CUpti_ActivityObjectKindId *id) {
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
getComputeApiKindString(CUpti_ActivityComputeApiKind kind) {
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
printActivity(CUpti_Activity *record) {
  switch (record->kind) {
  case CUPTI_ACTIVITY_KIND_DEVICE: {
    CUpti_ActivityDevice2 *device = (CUpti_ActivityDevice2 *)record;
    printf("DEVICE %s (%u), capability %u.%u, global memory (bandwidth %u GB/s, size %u MB), "
           "multiprocessors %u, clock %u MHz\n",
           device->name, device->id,
           device->computeCapabilityMajor, device->computeCapabilityMinor,
           (unsigned int)(device->globalMemoryBandwidth / 1024 / 1024),
           (unsigned int)(device->globalMemorySize / 1024 / 1024),
           device->numMultiprocessors, (unsigned int)(device->coreClockRate / 1000));
    break;
  }
  case CUPTI_ACTIVITY_KIND_DEVICE_ATTRIBUTE: {
    CUpti_ActivityDeviceAttribute *attribute = (CUpti_ActivityDeviceAttribute *)record;
    printf("DEVICE_ATTRIBUTE %u, device %u, value=0x%llx\n",
           attribute->attribute.cupti, attribute->deviceId, (unsigned long long)attribute->value.vUint64);
    break;
  }
  case CUPTI_ACTIVITY_KIND_CONTEXT: {
    CUpti_ActivityContext *context = (CUpti_ActivityContext *)record;
    printf("CONTEXT %u, device %u, compute API %s, NULL stream %d\n",
           context->contextId, context->deviceId,
           getComputeApiKindString((CUpti_ActivityComputeApiKind)context->computeApiKind),
           (int)context->nullStreamId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMCPY: {
    CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *)record;
    printf("MEMCPY %s [ %llu - %llu ] device %u, context %u, stream %u, correlation %u/r%u\n",
           getMemcpyKindString((CUpti_ActivityMemcpyKind)memcpy->copyKind),
           (unsigned long long)(memcpy->start - startTimestamp),
           (unsigned long long)(memcpy->end - startTimestamp),
           memcpy->deviceId, memcpy->contextId, memcpy->streamId,
           memcpy->correlationId, memcpy->runtimeCorrelationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MEMSET: {
    CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *)record;
    printf("MEMSET value=%u [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
           memset->value,
           (unsigned long long)(memset->start - startTimestamp),
           (unsigned long long)(memset->end - startTimestamp),
           memset->deviceId, memset->contextId, memset->streamId,
           memset->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_KERNEL:
  case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
    const char *kindString = (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) ? "KERNEL" : "CONC KERNEL";
    CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
    printf("%s \"%s\" [ %llu - %llu ] device %u, context %u, stream %u, correlation %u\n",
           kindString,
           kernel->name,
           (unsigned long long)(kernel->start - startTimestamp),
           (unsigned long long)(kernel->end - startTimestamp),
           kernel->deviceId, kernel->contextId, kernel->streamId,
           kernel->correlationId);
    printf("    grid [%u,%u,%u], block [%u,%u,%u], shared memory (static %u, dynamic %u)\n",
           kernel->gridX, kernel->gridY, kernel->gridZ,
           kernel->blockX, kernel->blockY, kernel->blockZ,
           kernel->staticSharedMemory, kernel->dynamicSharedMemory);
    break;
  }
  case CUPTI_ACTIVITY_KIND_DRIVER: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("DRIVER cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
           api->cbid,
           (unsigned long long)(api->start - startTimestamp),
           (unsigned long long)(api->end - startTimestamp),
           api->processId, api->threadId, api->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_RUNTIME: {
    CUpti_ActivityAPI *api = (CUpti_ActivityAPI *)record;
    printf("RUNTIME cbid=%u [ %llu - %llu ] process %u, thread %u, correlation %u\n",
           api->cbid,
           (unsigned long long)(api->start - startTimestamp),
           (unsigned long long)(api->end - startTimestamp),
           api->processId, api->threadId, api->correlationId);
    break;
  }
  case CUPTI_ACTIVITY_KIND_NAME: {
    CUpti_ActivityName *name = (CUpti_ActivityName *)record;
    switch (name->objectKind) {
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
  case CUPTI_ACTIVITY_KIND_MARKER: {
    CUpti_ActivityMarker2 *marker = (CUpti_ActivityMarker2 *)record;
    printf("MARKER id %u [ %llu ], name %s, domain %s\n",
           marker->id, (unsigned long long)marker->timestamp, marker->name, marker->domain);
    break;
  }
  case CUPTI_ACTIVITY_KIND_MARKER_DATA: {
    CUpti_ActivityMarkerData *marker = (CUpti_ActivityMarkerData *)record;
    printf("MARKER_DATA id %u, color 0x%x, category %u, payload %llu/%f\n",
           marker->id, marker->color, marker->category,
           (unsigned long long)marker->payload.metricValueUint64,
           marker->payload.metricValueDouble);
    break;
  }
  case CUPTI_ACTIVITY_KIND_OVERHEAD: {
    CUpti_ActivityOverhead *overhead = (CUpti_ActivityOverhead *)record;
    printf("OVERHEAD %s [ %llu, %llu ] %s id %u\n",
           getActivityOverheadKindString(overhead->overheadKind),
           (unsigned long long)overhead->start - startTimestamp,
           (unsigned long long)overhead->end - startTimestamp,
           getActivityObjectKindString(overhead->objectKind),
           getActivityObjectKindId(overhead->objectKind, &overhead->objectId));
    break;
  }
  default:
    printf("  <unknown>\n");
    break;
  }
}

static void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
  uint8_t *bfr = (uint8_t *)malloc(BUF_SIZE + ALIGN_SIZE);
  if (bfr == NULL) {
    printf("Error: out of memory\n");
    exit(-1);
  }

  *size = BUF_SIZE;
  *buffer = ALIGN_BUFFER(bfr, ALIGN_SIZE);
  *maxNumRecords = 0;
}

static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize);



void finiTrace() {
  CUPTI_CALL(cuptiActivityFlushAll(1));
}
#endif


using namespace std::chrono;



#ifdef _MSC_VER
//__declspec(selectany)
#else
__attribute__((weak))
#endif
void run(State &state, int pbsize);



class BenchmarkRun {
  friend class Rosetta;
  friend class dyn_array_base;

private:
  std::vector<IterationMeasurement> measurements;
  std::chrono::steady_clock::time_point startTime;
  std::chrono::steady_clock::time_point firstIterTime;


  // TODO: Running sum, sumsquare, mean(?) of exit-determining measurement

  bool verify;
  int exactRepeats = -1;
  std::filesystem::path verifyoutpath;
  std::chrono::seconds max_duration;

public:
  std::ofstream verifyout;

public:
  bool isVerifyRun() const { return verify; }
  bool isBenchRun() const { return !verify; }

public:
  explicit BenchmarkRun(bool verify, int exactRepeats, std::filesystem::path verifyoutpath, std::chrono::seconds max_duration) : verify(verify), exactRepeats(exactRepeats), verifyoutpath(verifyoutpath), max_duration(max_duration) {}

  void run(std::string program, int n) {
    startTime = std::chrono::steady_clock::now();

    if (!verifyoutpath.empty()) {
      verifyout.open(verifyoutpath, std::ios::trunc);
      //    assert(verifyout);
      assert(verifyout.good());
      assert(verifyout.is_open());
    }

    {
      State state{this};

#if ROSETTA_PLATFORM_NVIDIA
      // printf("## Platform_NVIDIA\n");
      //  TODO: exclude cupti time from startTime

      // subscribe to CUPTI callbacks
      // CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &trace));

      // DRIVER_API_CALL(cuInit(0));
      // CUcontext context = 0;
      //   CUdevice device = 0;
      //  DRIVER_API_CALL(cuCtxCreate(&context, 0, device));


      // CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
      // CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

      size_t attrValue = 0, attrValueSize = sizeof(size_t);
      // Device activity record is created when CUDA initializes, so we
      // want to enable it before cuInit() or any CUDA runtime call.
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
      // Enable all other activity record kinds.
      // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
      // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
      // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
      // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
      CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
      // CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

      // Register callbacks for buffer requests and for buffers completed by CUPTI.
      CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));


      uint64_t cuptiStart;
      CUPTI_CALL(cuptiGetTimestamp(&cuptiStart));
      startTimestamp = cuptiStart;
#endif



#if ROSETTA_PPM_OPENMP_TARGET
#if defined(__clang__) || (defined(__GNUC__) && (__GNUC__ >= 12))
      auto default_device = omp_get_default_device();
      auto num_devices = omp_get_num_devices();
      auto cur_device = omp_get_device_num();
      std::cout << "OpenMP Target default device = " << default_device << ", current device = " << cur_device << "  of " << num_devices << " devices\n";
#endif
      // auto numProps =  omp_get_num_interop_properties(omp_ipr_fr_id);
#endif

#if ROSETTA_PPM_OPENMP || ROSETTA_PLATFORM_OPENMP
      auto tickduration = omp_get_wtick();
      std::cout << "OpenMP Wall tick = " << tickduration << "\n";
#endif



      preinitRSS = getMaxRSS();

      // TODO: make flexible, this is just on way to find a run function
      ::run(state, n);

      lastRSS = getMaxRSS();
    }

    if (verifyout.is_open()) {
      std::cerr << "Closing verifyfile: " << verifyoutpath << std::endl;
      verifyout.close();
    }


#if ROSETTA_PLATFORM_NVIDIA
    cuptiFinalize();
#endif


    assert(!curAllocatedBytes && "Should not leak memory");

    // TODO: print basic timing info so user knows something happened
  }



  bool started = false;


  // TODO: Per-iteration data in separate object?
  std::chrono::high_resolution_clock::time_point startWall;
  usage_duration_t startUser;   // in seconds; TODO: use native type
  usage_duration_t startKernel; // in seconds; TODO: use native type
  double startOmpWtime;         // in seconds

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


  memory_t curAllocatedBytes = 0;
  memory_t peakAllocatedBytes = 0;

  memory_t preinitRSS = 0;
  memory_t postinitRSS = 0;
  memory_t firstRSS = 0;
  memory_t lastRSS = 0;

  void start() {
    assert(!started && "Must not interleave interation scope");
    started = true;


#ifdef ROSETTA_PLATFORM_NVIDIA
    cuptiStartHtoD = std::numeric_limits<uint64_t>::max();
    cuptiStopHtoD = std::numeric_limits<uint64_t>::min();
    cuptiStartDtoH = std::numeric_limits<uint64_t>::max();
    cuptiStopDtoH = std::numeric_limits<uint64_t>::min();
    cuptiStartCompute = std::numeric_limits<uint64_t>::max();
    cuptiStopCompute = std::numeric_limits<uint64_t>::min();
    cuptiStartOther = std::numeric_limits<uint64_t>::max();
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

#if ROSETTA_PPM_OPENMP || ROSETTA_PLATFORM_OPENMP
    startOmpWtime = omp_get_wtime();
#endif

#if ROSETTA_PPM_CUDA
    cudaEventRecord(startCuda);
#endif
  }

  void stop() {
    assert(started && "No iteration active?");


    benchmark::ClobberMemory(); // even necessary?


#if ROSETTA_PPM_CUDA || ROSETTA_PLATFORM_NVIDIA
    BENCH_CUDA_TRY(cudaDeviceSynchronize());
#endif

#if ROSETTA_PPM_CUDA
    BENCH_CUDA_TRY(cudaEventRecord(stopCuda));
#endif


    auto stopWall = std::chrono::high_resolution_clock::now();
    auto [stopUser, stopKernel] = ProcessCPUUsage();

#if ROSETTA_PPM_OPENMP || ROSETTA_PLATFORM_OPENMP
    auto stopOmpWtime = omp_get_wtime();
    auto durationOmpWtime = stopOmpWtime - startOmpWtime;
#endif


#if ROSETTA_PPM_CUDA
    BENCH_CUDA_TRY(cudaEventRecord(stopCuda));
#endif

    auto durationWall = stopWall - startWall;
    auto durationUser = stopUser - startUser;
    auto durationKernel = stopKernel - startKernel;
#if ROSETTA_PPM_CUDA
    float durationCuda; // ms
    BENCH_CUDA_TRY(cudaEventSynchronize(stopCuda));
    BENCH_CUDA_TRY(cudaEventElapsedTime(&durationCuda, startCuda, stopCuda));
    BENCH_CUDA_TRY(cudaEventDestroy(startCuda));
    BENCH_CUDA_TRY(cudaEventDestroy(stopCuda));
#else
    float durationCuda = 0;
#endif

    durationWall = std::max(decltype(durationWall)::zero(), durationWall);
    // durationUser = std::max(0.0, durationUser);
    // durationKernel = std::max(0.0, durationKernel);

#if ROSETTA_PLATFORM_NVIDIA
    CUPTI_CALL(cuptiActivityFlushAll(1));
#endif


    started = false;



#if ROSETTA_PLATFORM_NVIDIA
    auto firstEvent = std::min({cuptiStartHtoD, cuptiStartDtoH, cuptiStartCompute, cuptiStartOther});
    auto lastEvent = std::max({cuptiStopHtoD, cuptiStopDtoH, cuptiStopCompute, cuptiStopOther});
#endif

    IterationMeasurement m;
    m.values[WallTime] = durationWall;
    m.values[UserTime] = durationUser;
    m.values[KernelTime] = durationKernel;
#if ROSETTA_PPM_OPENMP || ROSETTA_PLATFORM_OPENMP
    m.values[OpenMPWTime] = std::chrono::duration<decltype(durationOmpWtime), std::chrono::seconds::period>(durationOmpWtime);
#endif
#if ROSETTA_PPM_CUDA
    m.values[AccelTime] = std::chrono::duration<decltype(durationCuda), std::chrono::milliseconds::period>(durationCuda);
#endif
#if ROSETTA_PLATFORM_NVIDIA
    if (firstEvent <= lastEvent)
      m.values[Cupti] = std::chrono::duration<uint64_t, std::chrono::nanoseconds::period>(lastEvent - firstEvent);
    if (cuptiStartCompute <= cuptiStopCompute)
      m.values[CuptiCompute] = std::chrono::duration<uint64_t, std::chrono::nanoseconds::period>(cuptiStopCompute - cuptiStartCompute);
    if (cuptiStartHtoD <= cuptiStopHtoD)
      m.values[CuptiTransferToDevice] = std::chrono::duration<uint64_t, std::chrono::nanoseconds::period>(cuptiStopHtoD - cuptiStartHtoD);
    if (cuptiStartDtoH <= cuptiStopDtoH)
      m.values[CuptiTransferToHost] = std::chrono::duration<uint64_t, std::chrono::nanoseconds::period>(cuptiStopDtoH - cuptiStartDtoH);
#endif



    measurements.push_back(std::move(m));
  }

  int computeMoreIterations() {
    // When verifying, always do exactly one iteration
    // TODO: supersede by exactRepeats
    if (verify)
      return 1 - measurements.size();



    // If the number of iterations was specified on the command line, use that
    if (exactRepeats >= 1)
      return exactRepeats - measurements.size();



    auto now = std::chrono::steady_clock::now();
    auto elapsed = now - startTime;
    auto avgDuration = std::chrono::duration<double>(to_seconds(now - firstIterTime) / measurements.size());

    // Estimate how many iterations we can do before max elapsed is exceeded.
    // Note: max_duration includes startup time (firstIterTime - startTime)
    // auto iters_to_elapse =  std::floor(  (max_duration -  (firstIterTime - startTime)  ) / avgDuration)  ;
    auto iters_before_elapse = std::floor((max_duration - elapsed) / avgDuration);
    size_t howManyMoreIterations = (iters_before_elapse <= 0) ? 0 : std::llround(iters_before_elapse);

    // If time has elapsed, no more iterations to do
    if (howManyMoreIterations == 0)
      return 0;

    // If we are stable enough, finish earlier.
    // Get at least 5 measurements before any statistical significance
    // FIXME: Isnt' that quite low? Make configurable
    if (measurements.size() >= 5) {
      // TODO: Remove cold runs?
      std::vector<double> vals;
      vals.reserve(measurements.size());
      for (auto &&m : measurements) {
        vals.push_back(to_seconds(m.values[WallTime]));
      }
      Statistic stat{vals.data(), vals.size()};

      // Stop if goal has been reached
      // Goal: 95%-CI within 0.01 == 1% of mean
      // TODO: Make configurable
      if (stat.relerr() <= 0.01)
        return 0;


      auto iters_until_stable = std::max<size_t>(stat.min_more_samples_rel(0.05), 1); // Note: the ratio is more optimistic here (90% instead 95%)
      howManyMoreIterations = std::min(howManyMoreIterations, iters_until_stable);
    } else {
      howManyMoreIterations = std::min(howManyMoreIterations, 5 - measurements.size());
    }

    return howManyMoreIterations;
  }


  int refresh() {
    switch (measurements.size()) {
    case 0:
      postinitRSS = getMaxRSS();
      firstIterTime = std::chrono::steady_clock::now();
      return 1; // Ensure that we can measure firstRSS (and emit verify data if enabled) after first iteration
    case 1:
      firstRSS = getMaxRSS();
      break;
    }

    auto howManyMoreIterations = computeMoreIterations();
    assert(howManyMoreIterations >= 0);
    measurements.reserve(measurements.size() + howManyMoreIterations);
    return howManyMoreIterations;
  }



#if ROSETTA_PLATFORM_NVIDIA
  /// FIXME: Called from other threads. Require Mutex?
  void handleCuptiActivity(CUpti_Activity *record) {
    // assert(started);
    //  Ignore events outside iteration
    if (!started) {
      printActivity(record);
      return;
    }

    switch (record->kind) {
    case CUPTI_ACTIVITY_KIND_KERNEL:
    case CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL: {
      CUpti_ActivityKernel5 *kernel = (CUpti_ActivityKernel5 *)record;
      cuptiStartCompute = std::min(cuptiStartCompute, kernel->start);
      cuptiStopCompute = std::max(cuptiStopCompute, kernel->end);
      break;
    }
    case CUPTI_ACTIVITY_KIND_MEMCPY: {
      // TODO: record amount of data transferred
      CUpti_ActivityMemcpy3 *memcpy = (CUpti_ActivityMemcpy3 *)record;
      auto copyKind = (CUpti_ActivityMemcpyKind)memcpy->copyKind;

      switch (copyKind) {
      case CUPTI_ACTIVITY_MEMCPY_KIND_HTOD:
        cuptiStartHtoD = std::min(cuptiStartHtoD, memcpy->start);
        cuptiStopHtoD = std::max(cuptiStopHtoD, memcpy->end);
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

    case CUPTI_ACTIVITY_KIND_MEMSET: {
      CUpti_ActivityMemset2 *memset = (CUpti_ActivityMemset2 *)record;
      cuptiStartOther = std::min(cuptiStartOther, memset->start);
      cuptiStopOther = std::max(cuptiStopOther, memset->end);
      break;
    }

    default:
      printActivity(record);
    }
  }
#endif
};



int State::refresh() {
  return impl->refresh();
}

void State::addAllocatedBytes(size_t size) {
  impl->curAllocatedBytes += size;
  impl->peakAllocatedBytes = std::max(impl->peakAllocatedBytes, impl->curAllocatedBytes);
}
void State::subAllocatedBytes(size_t size) {
  assert(size <= impl->curAllocatedBytes);
  impl->curAllocatedBytes -= size;
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
  // 7. Avoid cancellation in gramschmidt
  for (ssize_t i = 0; i < count; i += 1)
    data[i] = 0.5 + i * 0.125;
}

template <typename T>
constexpr int log10ceil(T num) {
  return num < 10 ? 1 : 1 + log10ceil(num / 10);
}

// TODO: Make template to also cover float
void DataHandler<double>::verify(double *data, ssize_t count, std::vector<size_t> dims, std::string_view name) {
  // bool warning_shown = false;
  for (ssize_t i = 0; i < count; i += 1) {
    auto val = data[i];
    if (std::isinf(val) || std::isnan(val)) {
      std::cerr << "WARNING: Inf/NaN output " << val << " at " << name << "[" << i << "]\n";
      break;
    }
  }

  if (!impl->isVerifyRun())
    return;

  auto &verifyout = impl->verifyout;

  verifyout << "array";     // Output kind
  verifyout << " binary64"; // Data format according to IEEE 754-2019

  verifyout << ' ' << dims.size();
  for (auto l : dims)
    verifyout << ' ' << l;

  if (!name.empty())
    verifyout << ' ' << name;

  verifyout << ':';


  for (ssize_t i = 0; i < count; i += 1) {
    auto val = data[i];

    // https://stackoverflow.com/a/68475665
    // Evaluates to 24
    constexpr int MaxConvLen = 4 +
                               std::numeric_limits<double>::max_digits10 +
                               std::max(2, log10ceil(std::numeric_limits<double>::max_exponent10));
    std::array<char, MaxConvLen> str;

#if HAS_CHARCONV_DOUBLE
    // std::to_chars returns a string representation that allows recovering the original float binary representation, and locale-independent.
    // FIXME: Does the python part actually do? Could also use std::chars_format::hex or use binary directly.
    // https://eel.is/c++draft/charconv.to.chars#2
    auto [ptr, ec] = std::to_chars(str.data(), str.data() + str.size(), val, std::chars_format::general);
    assert(ec == std::errc());
    / std::string_view strv(str.data(), ptr - str.data());
#else
    auto strv = std::to_string(val);
#endif

    verifyout << '\t' << strv;
  }
  verifyout << '\n';
}


void Iteration::start() {
  state.impl->start();
}

void Iteration::stop() {
  // TODO: save indirection
  state.impl->stop();
}



#if ROSETTA_PLATFORM_NVIDIA
static CUpti_SubscriberHandle subscriber;
static RuntimeApiTrace_t trace[LAUNCH_LAST];
#endif


static const char *getUnit(Measure measure) {
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


template <typename... LAMBDAS>
struct visitors : LAMBDAS... {
  using LAMBDAS::operator()...;
};

template <typename... LAMBDAS>
visitors(LAMBDAS... x) -> visitors<LAMBDAS...>;
template <class... T>
constexpr bool always_false = false;

struct duration_formatter {
  std::ostringstream &buf;

  template <typename T>
  typename std::enable_if<std::is_floating_point<T>::value>::type printNumber(T x) {
    buf << std::setprecision(std::numeric_limits<double>::digits10 + 1) << x;
  }

  template <typename T>
  typename std::enable_if<std::is_integral<T>::value>::type printNumber(T x) {
    buf << x;
  }

  template <typename Ratio>
  void printUnit() {}

  template <typename T, typename Ratio>
  void operator()(std::chrono::duration<T, Ratio> v) {
    printNumber(v.count());

    if constexpr (std::is_same_v<Ratio, std::chrono::seconds::period>) {
      buf << " s";
    } else if constexpr (std::is_same_v<Ratio, std::chrono::nanoseconds::period>) {
      buf << " ns";
    } else if constexpr (std::is_same_v<Ratio, std::chrono::milliseconds::period>) {
      buf << " ms";
    } else {
      static_assert(always_false<T>, "Unhandled time unit");
    }
  }

  template <typename T>
  void operator()(std::chrono::duration<T, usage_duration_t::period> v) {
    // fallback to nanoseconds
    return this->operator()(std::chrono::duration_cast<std::chrono::nanoseconds>(v));
  }


  void operator()(std::monostate) {}
};



static std::string formatDuration(duration_t duration) {
  std::ostringstream buf;
  if (std::holds_alternative<std::monostate>(duration))
    return ""; // Should not have been called

  duration_formatter callme{buf};
  std::visit(callme, duration);

  return buf.str();
}

extern const char *rosetta_default_results_dir;
extern const char *rosetta_configname;



template <typename T, typename Y>
static duration_t internal_add(const T &lhs, const Y &rhs) {
  if constexpr (std::is_same_v<T, std::monostate>) {
    return rhs;
  } else if constexpr (std::is_same_v<T, std::monostate>) {
    return lhs;
  } else {
    if constexpr (std::is_same_v<T, Y>) {
      return lhs + rhs;
    }
    using CommonTy = std::common_type<T, Y>;
    // TODO: std::common_type, but requires all combinations of T,Y to be supported
    // TODO: fall back to double/seconds (lower common denominator)
    // return lhs + rhs;
  }
  abort();
}

static duration_t operator+(const duration_t &lhs, const duration_t &rhs) {
  return std::visit(
      [&rhs](const auto &lhs) {
        return std::visit(
            [&lhs](const auto &rhs) {
              return internal_add(lhs, rhs);
            },
            rhs);
      },
      lhs);
}

static const duration_t &operator+=(duration_t &lhs, const duration_t &rhs) {
  lhs = lhs + rhs;
  return lhs;
}


const char *getMeasureDesc(Measure m) {
  switch (m) {
  case WallTime:
    return "Wall time";
  case UserTime:
    return "User time";
  case KernelTime:
    return "Kernel time";
  case OpenMPWTime:
    return "OpenMP time";
  case AccelTime:
    return "CUDA Event time";
  case Cupti:
    return "Nvprof total time";
  case CuptiCompute:
    return "Nvprof compute time";
  case CuptiTransferToDevice:
    return "Nvprof H->D time";
  case CuptiTransferToHost:
    return "Nvprof D->H time";
  }
  return "<measure description missing>";
}


extern const char *bench_name;
extern int64_t bench_default_problemsize;
extern const char *bench_buildtype;


// TODO: make singleton?
struct Rosetta {
  static constexpr const char *measureDesc[MeasureCount] = {"Wall Clock", "User", "Kernel", "GPU"};
  static constexpr const char *measureName[MeasureCount] = {"walltime", "usertime", "kerneltime", "ompwtime", "acceltime", "cupti", "cupti_compute", "cupti_todev", "cupti_fromdev"};

  static std::string escape(std::string s) {
    return s; // TODO
  }

  static BenchmarkRun *currentRun;
  static void run(std::filesystem::path executable, std::string program, std::filesystem::path xmlout, bool verify, std::filesystem::path verifyout, int n, int repeats, std::string timestamp, std::chrono::seconds max_duration) {
    BenchmarkRun executor(verify, repeats, verifyout, max_duration);
    currentRun = &executor;
    executor.run(program, n);

    if (verify) {
    } else {
      int numMeasures = executor.measurements.size();
      int startMeasures = 0;
      if (numMeasures >= 2 && repeats == -1) {
        // Remove cold run if we can afford it and no option to omit was given.
        startMeasures = 1;
      }

#if ROSETTA_PPM_SERIAL
      const char *ppm_variant = "serial";
#endif
#if ROSETTA_PPM_CUDA
      const char *ppm_variant = "cuda";
#endif
#if ROSETTA_PPM_OPENMP_PARALLEL
      const char *ppm_variant = "openmp-parallel";
#endif
#if ROSETTA_PPM_OPENMP_TASK
      const char *ppm_variant = "openmp-task";
#endif
#if ROSETTA_PPM_OPENMP_TARGET
      const char *ppm_variant = "openmp-target";
#endif

#if ROSETTA_PPM_SYCL
      const char *ppm_variant = "sycl";
#endif

      std::cout << "Benchmarking done, " << numMeasures << " measurements recorded.\n";

      duration_t sum[MeasureCount];
      for (int i = startMeasures; i < numMeasures; i += 1) {
        for (int k = 0; k <= MeasureLast; k += 1) {
          auto &&m = executor.measurements[i];
          auto &&val = m.values[k];
          sum[k] += val;
        }
      }

      // TODO: exclude cold iterations
      std::cout << "Average (arithmetic mean) times:\n";
      for (int k = 0; k <= MeasureLast; k += 1) { // TODO: make an iterator over all measures to make such a loop nices (and make Measure an enum class)
        auto &&thesum = sum[k];
        if (std::holds_alternative<std::monostate>(thesum))
          continue;
        auto avg = to_seconds(thesum) / numMeasures;

        // TODO: Scale avg to ns/us/ms/s/m/h as needed
        std::cout << getMeasureDesc((Measure)k) << ": " << avg << "s\n";
      }



      if (!xmlout.empty()) {
        std::cout << "Writing result file: " << xmlout << "\n";
        std::filesystem::create_directories(xmlout.parent_path());
        std::ofstream cxml(xmlout, std::ios::trunc);
        if (!cxml.good())
          perror("Unable to open xmlout file");
        assert(cxml.good());
        assert(cxml.is_open());

        cxml << R"(<?xml version="1.0" encoding="UTF-8" ?>)" << std::endl;
        cxml << R"(<benchmarks>)" << std::endl;
        cxml << R"(  <benchmark name=")" << escape(program) << R"(" n=")" << n << "\" cold_iterations=\"" << startMeasures << "\" peak_alloc=\"" << executor.peakAllocatedBytes << "\" ppm=\"" << ppm_variant << '\"';
        if (!timestamp.empty())
          cxml << " timestamp=\"" << escape(timestamp) << '\"';
        if (strlen(rosetta_configname) >= 1)
          cxml << " configname=\"" << escape(rosetta_configname) << '\"';
        if (strlen(bench_buildtype) >= 1)
          cxml << " buildtype=\"" << escape(bench_buildtype) << '\"';
        cxml << " maxrss=\"" << executor.lastRSS - executor.preinitRSS << "\""; // Discount RSS for process overhead to avoid a OS-specific constant baseline (not entirely accurate; those parts like the loader may be removed from the working set during exeuction because of memory pressure)
        cxml << ">" << std::endl;
        for (int i = startMeasures; i < numMeasures; i += 1) {
          auto &m = executor.measurements[i];
          // for (auto &m :executor. measurements) {
          // TODO: custom times and units
          cxml << "    <iteration";

          for (int i = 0; i <= MeasureLast; i += 1) {
            auto measure = (Measure)i;
            auto &val = m.values[measure];
            if (std::holds_alternative<std::monostate>(val))
              continue;
            cxml << ' ' << measureName[measure] << "=\"" << formatDuration(m.values[measure]) << '\"';
          }

          cxml << R"(/>)" << std::endl;
        }
        // TODO: run properties: num threads, device, executable hash, allocated bytes, num flop (calculated), num updates, performance counters, ..
        cxml << R"(  </benchmark>)" << std::endl;
        cxml << R"(</benchmarks>)" << std::endl;
        cxml.close();
        std::cout << "Done writing to file: " << xmlout << "\n";
      }
    }

    currentRun = nullptr;
  }


#if ROSETTA_PLATFORM_NVIDIA
  static void handleCuptiActivity(CUpti_Activity *record) {
    if (!currentRun) {
      // Activity not asociated with a run
      return;
    }


    currentRun->handleCuptiActivity(record);
  }
#endif
};


BenchmarkRun *Rosetta::currentRun = nullptr;



#if ROSETTA_PLATFORM_NVIDIA
static void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize) {
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        Rosetta::handleCuptiActivity(record);
      } else if (status == CUPTI_ERROR_MAX_LIMIT_REACHED)
        break;
      else {
        CUPTI_CALL(status);
      }
    } while (1);

    // report any records dropped from the queue
    size_t dropped;
    CUPTI_CALL(cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped));
    if (dropped != 0) {
      printf("Dropped %u activity records\n", (unsigned int)dropped);
    }
  }

  free(buffer);
}
#endif



std::tuple<std::string_view, std::optional<std::string_view>> nextArg(int argc, char *argv[], int &i) {
  std::string_view arg{argv[i]};
  i += 1;
  if (arg.length() >= 3 && arg.substr(0, 2) == "--" && isalpha(arg[2])) {
    std::string_view sw = arg.substr(2);
    auto eqpos = sw.find('=');
    if (eqpos != std::string_view::npos) {
      auto swname = sw.substr(0, eqpos);
      auto swarg = sw.substr(eqpos + 1);
      return {swname, swarg};
    }
    return {sw, {}};
  }
  if (arg.length() >= 2 && arg[0] == '-' && isalpha(arg[1])) {
    auto swname = arg.substr(1, 1);
    if (arg.length() > 2) {
      auto swarg = arg.substr(2);
      return {swname, swarg};
    }
    return {swname, {}};
  }

  // Not recognized as a switch. Assume positional argument.
  return {std::string_view(), arg};
}


static std::string_view trim(std::string_view str) {
  constexpr char const *whitespace{" \t\r\n"};
  auto start = str.find_first_not_of(whitespace);
  if (start == std::string::npos)
    return ""; // contains only whitespace chars
  auto stop = str.find_last_not_of(whitespace, str.length());
  if (stop == std::string::npos)
    return ""; // contains only whitespace chars
  if (stop < start)
    return "";
  return str.substr(start, stop - start + 1);
}


static int parseInt(std::string_view s) {
#if HAS_INCLUDE_CHARCONV
  int result;
  auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.length(), result);
  // TODO: check for error
  return result;
#else
  std::string str(s);
  return std::stoi(str);
#endif
}


dyn_array_base::
    dyn_array_base(BenchmarkRun *impl, int size, bool verify, std::vector<size_t> dims, std::string_view name) : impl(impl), size(size), verify(verify), dims(std::move(dims)), name(name) {
  if (!impl)
    return;
  impl->curAllocatedBytes += size;
  impl->peakAllocatedBytes = std::max(impl->peakAllocatedBytes, impl->curAllocatedBytes);
}


dyn_array_base::
    ~dyn_array_base() {
  if (!impl)
    return;
  assert(size <= impl->curAllocatedBytes);
  impl->curAllocatedBytes -= size;
}



// TODO: do this randomly once between benchmarking?
static void warn_load() {
  // Ensure lazy symbol resolution has happened
  ProcessCPUUsage();
  std::chrono::steady_clock::now();


  // Do 100ms of computation and see whether user time agrees
  auto [startUser, startKernel] = ProcessCPUUsage();
  auto start = std::chrono::steady_clock::now();
  auto wtime = start - start;
  while (true) {
    wtime = std::chrono::steady_clock::now() - start;
    if (wtime >= 100ms)
      break;
  }
  auto [stopUser, stopKerne] = ProcessCPUUsage();
  auto utime = stopUser - startUser;

  // 2ms noise allowance
  if (wtime - 2ms > utime) {
    std::cerr << "WARNING: Additional load detected during benchmarking (computed for " << std::chrono::duration_cast<std::chrono::milliseconds>(wtime).count() << "ms but only " << std::chrono::duration_cast<std::chrono::milliseconds>(utime).count() << "ms was available to this process)\n";
  }
}



int main(int argc, char *argv[]) {
  assert(argc >= 1);

  std::filesystem ::path program(argv[0]);
  std::string benchname;
  if (bench_name) {
    // preferably use program name linked into the program
    benchname = bench_name;
    // printf("bench_name %s\n", bench_name);
  } else {
    // fallback to executable name
    auto progname = program.filename().string();

    auto dotpos = progname.find_first_of('.');
    benchname = progname;
    if (dotpos != std::string::npos)
      benchname = progname.substr(0, dotpos);
  }

  std::string_view problemsizefile; // TOOD: Use std::filesystem::path
  std::string_view verifyfile;
  std::string_view xmlout;

  // Use current time as timestamp by default
  auto now = std::chrono::system_clock::now();
  auto now_tm = std::chrono::system_clock::to_time_t(now);
  std::stringstream ss;
  ss << std::put_time(std::localtime(&now_tm), "%F %T%z");
  std::string timestamp = ss.str();

  // TODO: Consider Catch2 for cmd parsing
  // using namespace Catch::Clara;

  int problemsize = -1;
  int repeats = -1;
  int cold = -1;
  int i = 1;
  bool verify = false;
  std::chrono::seconds max_duration = 10s;
  while (i < argc) {
    auto [name, val] = nextArg(argc, argv, i);

    if (name == "n" || name == "problemsize" || name == "pbsize") {
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      problemsize = parseInt(*val);
    } else if (name == "problemsizefile") {
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      problemsizefile = *val;
    } else if (name == "repeats") {
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      repeats = parseInt(*val);
    } else if (name == "verify") {
      assert(!val.has_value());
      verify = true;
    } else if (name == "verifyfile") {
      assert(val.has_value() || i <= argc);
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      verifyfile = *val;
    } else if (name == "xmlout") {
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      xmlout = *val;
    } else if (name == "timestamp") {
      // Timestamp to log when the benchmarking was invoked,
      // So all benchmark get the same timestamp
      assert(val.has_value() || i <= argc);
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      timestamp = *val;
    } else if (name == "max-duration") {
      assert(val.has_value() || i <= argc);
      if (!val.has_value() && i <= argc) {
        val = argv[i];
        i += 1;
      }
      max_duration = std::chrono::seconds(parseInt(*val));
    } else {
      assert(!"unknown switch");
    }
  }


  int64_t n = -1;
  if (problemsize >= 0) {
    // Explicit problemsize has priority
    n = problemsize;
  } else if (!problemsizefile.empty()) {
    std::vector<std::string> lines;
    {
      std::ifstream psfile(std::string(problemsizefile).c_str(), std::ios::in);
      std::string myline;
      for (std::string myline; std::getline(psfile, myline);)
        lines.push_back(myline);
    }

    std::vector<std::string> seclines;
    int i = 0;
    auto secheader = "[" + benchname + "]";
    while (i < lines.size()) {
      if (trim(lines[i]) == secheader) {
        i += 1;
        while (i < lines.size()) {
          auto trimmed = trim(lines[i]);
          if (trimmed.substr(0, 1) == "[")
            break;
          if (!trimmed.empty())
            seclines.push_back(std::string(trimmed));
          i += 1;
        }
        break;
      }
      i += 1;
    }

    for (auto &&line : seclines) {
      auto lineview = std::string_view(line);
      auto eqpos = line.find_first_of('=');
      if (eqpos == std::string::npos)
        continue;
      auto key = trim(lineview.substr(0, eqpos));
      auto val = trim(lineview.substr(eqpos + 1));

      if (key == "n") {
        auto nval = parseInt(val);
        n = nval;
        continue;
      }
    }
  } else {
    n = bench_default_problemsize;
  }



  std::filesystem::path resultsfilename;
  if (!xmlout.empty()) {
    resultsfilename = xmlout;
  } else {
    // Generate unique filename. Keep in sync with rosetta.py
    // $ROSETTA_RESULTS_DIR/$benchname/{datetime}.xml
    if (strlen(rosetta_default_results_dir) > 0) {
      resultsfilename = rosetta_default_results_dir;
      resultsfilename /= benchname;
    } else {
      // Use cwd, without subdirs
    }

    // TODO: Use timestamp for consistency
    std::string suffix;
    int i = 0;
    while (true) {
      auto now = std::chrono::system_clock::now();
      std::time_t now_time = std::chrono::system_clock::to_time_t(now);
      auto q = std::ctime(&now_time);
      char buf[200];
      std::strftime(buf, sizeof(buf), "%Y%m%d_%H%M", std::localtime(&now_time));
      std::string filename = std::string(buf) + "_" + benchname + suffix + ".xml";

      // auto filename = std::vformat("{0:%F_%R}_{1}{2}.xml",now,benchname, suffix  );
      resultsfilename /= filename;
      if (!std::filesystem::exists(resultsfilename))
        break;
      i += 1;
      suffix = "_" + std::to_string(i);
    }
  }

  // TOOD: allow more than one benchmark per executable
  assert(n >= 1);

  if (!verify)
    warn_load();
  Rosetta::run(program, benchname, resultsfilename, verify, verifyfile, n, repeats, timestamp, max_duration);
  if (!verify)
    warn_load();

  return EXIT_SUCCESS;
}
