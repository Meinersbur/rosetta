#include "rosetta.h"
#include <cassert>
#include <algorithm>
#include <numeric>
#include <iostream>


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


#include "check.h"
#include "log.h"
#include "sleep.h"
#include "string_util.h"
// Suppress unused warnings on helper functions.
#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wunused-function"
#endif

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



#if 0
std::pair<double,double> ThreadCPUUsage() {
#if defined(BENCHMARK_OS_WINDOWS)
    HANDLE this_thread = GetCurrentThread();
    FILETIME creation_time;
    FILETIME exit_time;
    FILETIME kernel_time;
    FILETIME user_time;
    GetThreadTimes(this_thread, &creation_time, &exit_time, &kernel_time,
        &user_time);
    return MakeTime(kernel_time, user_time);
#elif defined(BENCHMARK_OS_MACOSX)
    // FIXME We want to use clock_gettime, but its not available in MacOS 10.11.
    // See https://github.com/google/benchmark/pull/292
    mach_msg_type_number_t count = THREAD_BASIC_INFO_COUNT;
    thread_basic_info_data_t info;
    mach_port_t thread = pthread_mach_thread_np(pthread_self());
    if (thread_info(thread, THREAD_BASIC_INFO,
        reinterpret_cast<thread_info_t>(&info),
        &count) == KERN_SUCCESS) {
        return MakeTime(info);
    }
    DiagnoseAndExit("ThreadCPUUsage() failed when evaluating thread_info");
#elif defined(BENCHMARK_OS_EMSCRIPTEN)
    // Emscripten doesn't support traditional threads
    return ProcessCPUUsage();
#elif defined(BENCHMARK_OS_RTEMS)
    // RTEMS doesn't support CLOCK_THREAD_CPUTIME_ID. See
    // https://github.com/RTEMS/rtems/blob/master/cpukit/posix/src/clockgettime.c
    return ProcessCPUUsage();
#elif defined(BENCHMARK_OS_SOLARIS)
    struct rusage ru;
    if (getrusage(RUSAGE_LWP, &ru) == 0) return MakeTime(ru);
    DiagnoseAndExit("getrusage(RUSAGE_LWP, ...) failed");
#elif defined(CLOCK_THREAD_CPUTIME_ID)
    struct timespec ts;
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) == 0) return MakeTime(ts);
    DiagnoseAndExit("clock_gettime(CLOCK_THREAD_CPUTIME_ID, ...) failed");
#else
#error Per-thread timing is not available on your system.
#endif
}
#endif


using namespace std::chrono;


void Iteration::start() {
    startWall = std::chrono::high_resolution_clock::now();
   std::tie(startUser, startKernel) = ProcessCPUUsage();
}



void Iteration::stop() {
    //TODO: Throw away warmup

   auto stopWall = std::chrono::high_resolution_clock::now();
   auto [stopUser, stopKernel] = ProcessCPUUsage();

   auto durationWall = stopWall-startWall;
   auto durationUser = stopUser - startUser;
   auto durationKernel = stopKernel - startKernel;

   durationWall = std::max(decltype(durationWall)::zero(), durationWall);
   durationUser = std::max(0.0, durationUser);
   durationKernel = std::max(0.0, durationKernel);

   IterationMeasurement m;
   m.values[WallTime] = std::chrono::duration<double>(durationWall).count();
   m.values[UserTime] = durationUser;
   m.values[KernelTime] = durationKernel;
   state.measurements.push_back( std::move (m) );
}





int State::refresh() {
    auto now = std::chrono::steady_clock::now();
    auto duration =  now - startTime;

    if (duration >= 1s) // TODO: configure, until stability, max/min number iterations, ...
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


struct  Rosetta {
    static constexpr const char* measureDesc[MeasureCount] = {"Wall Clock", "User", "Kernel", "GPU"};

    static std::string  escape(std::string s) {
        return s; // TODO
    }

        static void run(const char *program, int n) {          
                std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock ::now();
                State state{startTime};

                ::run(state, n);
            
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

