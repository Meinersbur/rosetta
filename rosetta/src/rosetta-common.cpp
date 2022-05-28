#include "rosetta.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>

#if ROSETTA_PLATFORM_NVIDIA
#include <cupti.h>
#include <stdio.h>
#include <cuda.h>
#include <cupti.h>
#endif


#ifdef BENCHMARK_OS_WINDOWS
#define NOMINMAX 1
#define WIN32_LEAN_AND_MEAN 1
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

namespace {
#if defined(BENCHMARK_OS_WINDOWS)
   std::pair<  double,double> MakeTime(FILETIME const& kernel_time, FILETIME const& user_time) {
        ULARGE_INTEGER kernel;
        ULARGE_INTEGER user;
        kernel.HighPart = kernel_time.dwHighDateTime;
        kernel.LowPart = kernel_time.dwLowDateTime;
        user.HighPart = user_time.dwHighDateTime;
        user.LowPart = user_time.dwLowDateTime;
        return {user.QuadPart* 1e-7, kernel.QuadPart* 1e-7};
          //  (static_cast<double>(kernel.QuadPart) +  static_cast<double>(user.QuadPart)) * 1e-7;
    }
#elif !defined(BENCHMARK_OS_FUCHSIA)
    std::pair<  double,double>  MakeTime(struct rusage const& ru) {
        return { static_cast<double>(ru.ru_utime.tv_sec) + static_cast<double>(ru.ru_utime.tv_usec) * 1e-6, static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6 };
      //  return (static_cast<double>(ru.ru_utime.tv_sec) +  static_cast<double>(ru.ru_utime.tv_usec) * 1e-6 + static_cast<double>(ru.ru_stime.tv_sec) + static_cast<double>(ru.ru_stime.tv_usec) * 1e-6);
    }
#endif
#if defined(BENCHMARK_OS_MACOSX)
    double MakeTime(thread_basic_info_data_t const& info) {
        return (static_cast<double>(info.user_time.seconds) +
            static_cast<double>(info.user_time.microseconds) * 1e-6 +
            static_cast<double>(info.system_time.seconds) +
            static_cast<double>(info.system_time.microseconds) * 1e-6);
    }
#endif
#if defined(CLOCK_PROCESS_CPUTIME_ID) || defined(CLOCK_THREAD_CPUTIME_ID)
    double MakeTime(struct timespec const& ts) {
        return ts.tv_sec + (static_cast<double>(ts.tv_nsec) * 1e-9);
    }
#endif

    BENCHMARK_NORETURN static void DiagnoseAndExit(const char* msg) {
        std::cerr << "ERROR: " << msg << std::endl;
        std::exit(EXIT_FAILURE);
    }

}  // end namespace

std::pair<double,double> ProcessCPUUsage() {
#if defined(BENCHMARK_OS_WINDOWS)
    HANDLE proc = GetCurrentProcess();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    if (GetProcessTimes(proc, &creation_time, &exit_time, &kernel_time,
        &user_time))
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

const char *
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

const char *
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

uint32_t
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

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords)
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

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId, uint8_t *buffer, size_t size, size_t validSize)
{
  CUptiResult status;
  CUpti_Activity *record = NULL;

  if (validSize > 0) {
    do {
      status = cuptiActivityGetNextRecord(buffer, validSize, &record);
      if (status == CUPTI_SUCCESS) {
        printActivity(record);
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

void
initTrace() {
  size_t attrValue = 0, attrValueSize = sizeof(size_t);
  // Device activity record is created when CUDA initializes, so we
  // want to enable it before cuInit() or any CUDA runtime call.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DEVICE));
  // Enable all other activity record kinds.
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONTEXT));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_NAME));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MARKER));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL));
  CUPTI_CALL(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_OVERHEAD));

  // Register callbacks for buffer requests and for buffers completed by CUPTI.
  CUPTI_CALL(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));


#if 0
  // Get and set activity attributes.
  // Attributes can be set by the CUPTI client to change behavior of the activity API.
  // Some attributes require to be set before any CUDA context is created to be effective,
  // e.g. to be applied to all device buffer allocations (see documentation).
  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_SIZE, &attrValueSize, &attrValue));

  CUPTI_CALL(cuptiActivityGetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
  printf("%s = %llu\n", "CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT", (long long unsigned)attrValue);
  attrValue *= 2;
  CUPTI_CALL(cuptiActivitySetAttribute(CUPTI_ACTIVITY_ATTR_DEVICE_BUFFER_POOL_LIMIT, &attrValueSize, &attrValue));
#endif


  CUPTI_CALL(cuptiGetTimestamp(&startTimestamp));
}

void finiTrace()
{
   CUPTI_CALL(cuptiActivityFlushAll(1));
}
#endif 


using namespace std::chrono;



void Iteration::start() {
//printf("start\n");

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
}



void Iteration::stop() {
    //TODO: Throw away warmup

#if ROSETTA_PPM_CUDA
  cudaEventRecord(stopCuda);
#endif

   auto stopWall = std::chrono::high_resolution_clock::now();
   auto [stopUser, stopKernel] = ProcessCPUUsage();

#if ROSETTA_PPM_CUDA
BENCH_CUDA_TRY(cudaEventRecord(stopCuda));
#endif

   auto durationWall = stopWall-startWall;
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
   durationUser = std::max(0.0, durationUser);
   durationKernel = std::max(0.0, durationKernel);

   IterationMeasurement m;
   m.values[WallTime] = std::chrono::duration<double>(durationWall).count();
   m.values[UserTime] = durationUser;
   m.values[KernelTime] = durationKernel;
   m.values[AccelTime] = durationCuda / 1000.0;
   state.measurements.push_back( std::move (m) );
}





int State::refresh() {
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
}








#ifdef _MSC_VER
//__declspec(selectany)
#else
__attribute__((weak))
#endif
void run(State& state, int n);


#if ROSETTA_PLATFORM_NVIDIA
 static  CUpti_SubscriberHandle subscriber;
  static RuntimeApiTrace_t trace[LAUNCH_LAST];
#endif 

struct  Rosetta {
    static constexpr const char* measureDesc[MeasureCount] = {"Wall Clock", "User", "Kernel", "GPU"};

    static std::string  escape(std::string s) {
        return s; // TODO
    }

        static void run(const char *program, int n) {       
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock ::now();
                State state{startTime};

#if ROSETTA_PLATFORM_NVIDIA
// TODO: exclude cupti time from startTime

    // subscribe to CUPTI callbacks
  // CUPTI_CALL(cuptiSubscribe(&subscriber, (CUpti_CallbackFunc)getTimestampCallback, &trace));

  //DRIVER_API_CALL(cuInit(0));
 //CUcontext context = 0;
 //  CUdevice device = 0;
 // DRIVER_API_CALL(cuCtxCreate(&context, 0, device));


   //CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API));
  // CUPTI_CALL(cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API));

  initTrace();
#endif 

::run(state, n);

#if ROSETTA_PLATFORM_NVIDIA
  // display timestamps collected in the callback
 // displayTimestamps(trace);

 finiTrace();

 // CUPTI_CALL(cuptiUnsubscribe(subscriber));

cuptiFinalize();
#endif
            
                auto count = state.measurements.size();
                double sums[MeasureCount] ;
                for (int i = 0; i < MeasureCount;  i+=1 ) {
                    double sum = 0;
                    for (auto &m : state.measurements) {
                          auto val  = m.values[i];
                          sum += val;
                    }
                    sums[i]  = sum;
                }

                std::cout << R"(<?xml version="1.0"?>)" <<std::endl;
                std::cout << R"(<benchmarks>)" <<std::endl;
                std::cout << R"(  <benchmark name=")" << escape(program) <<   R"(" n=")" << n << R"(">)"<<std::endl;
                for (auto &m : state.measurements) {
                    // TODO: custom times and units
                    std::cout << R"(    <iteration walltime=")" << m.values[WallTime] << R"(" usertime=")" << m.values[UserTime] << R"(" kerneltime=")" << m.values[KernelTime] << R"(" acceltime=")" << m.values[AccelTime] << R"("/>)"<<std::endl;  
                }
                // TODO: run properties: num threads, device, executable hash, allocated bytes, num flop (calculated), num updates, performance counters, ..
                std::cout << R"(  </benchmark>)" <<std::endl;
                std::cout << R"(</benchmarks>)" <<std::endl;
        }
};




int main(int argc, char* argv[]) {
    auto program = argv[0];
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




    Rosetta::run( program, n );




    return EXIT_SUCCESS;
}

