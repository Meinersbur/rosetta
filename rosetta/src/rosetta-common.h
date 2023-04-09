#ifndef ROSETTA_COMMON_H
#define ROSETTA_COMMON_H

#include "internal_macros.h"




#ifdef BENCHMARK_OS_WINDOWS

#define NOMINMAX 1
#ifndef WIN32_LEAN_AND_MEAN // Already set by cupti
#define WIN32_LEAN_AND_MEAN 1
#endif
#include <shlwapi.h>
#undef StrCat // Don't let StrCat in string_util.h be renamed to lstrcatA
#include <Psapi.h>
#include <Psapi.h>  // memory_info(), memory_maps()
#include <bcrypt.h> // NTSTATUS
#include <ntstatus.h>
#include <signal.h>
#include <tlhelp32.h>
#include <versionhelpers.h>
#include <windows.h>

// #include <ntdef.h>
// #include <ntifs.h>
// (requires driver SDK)
#define NtCurrentProcess() ((HANDLE)(LONG_PTR)-1)
#define NT_SUCCESS(Status) (((NTSTATUS)(Status)) >= 0)
NTSTATUS(NTAPI *_NtQueryVirtualMemory)
(
	HANDLE ProcessHandle,
	PVOID BaseAddress,
	int MemoryInformationClass,
	PVOID MemoryInformation,
	SIZE_T MemoryInformationLength,
	PSIZE_T ReturnLength);
#define NtQueryVirtualMemory _NtQueryVirtualMemory
#define MemoryWorkingSetInformation 0x1
typedef struct _MEMORY_WORKING_SET_BLOCK {
	ULONG_PTR Protection : 5;
	ULONG_PTR ShareCount : 3;
	ULONG_PTR Shared : 1;
	ULONG_PTR Node : 3;
#ifdef _WIN64
	ULONG_PTR VirtualPage : 52;
#else
	ULONG VirtualPage : 20;
#endif
} MEMORY_WORKING_SET_BLOCK, *PMEMORY_WORKING_SET_BLOCK;
typedef struct _MEMORY_WORKING_SET_INFORMATION {
	ULONG_PTR NumberOfEntries;
	MEMORY_WORKING_SET_BLOCK WorkingSetInfo[1];
} MEMORY_WORKING_SET_INFORMATION, *PMEMORY_WORKING_SET_INFORMATION;

#else
#include <fcntl.h>
#ifndef BENCHMARK_OS_FUCHSIA
#include <sys/resource.h>
#endif
#include <sys/time.h>
#include <sys/types.h> // this header must be included before 'sys/sysctl.h' to avoid compilation error on FreeBSD
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

struct Rosetta;




namespace rosetta {




#if defined(BENCHMARK_OS_WINDOWS)
	using usage_duration_t = std::chrono::duration<ULONGLONG, std::ratio_multiply<std::chrono::nanoseconds::period, std::ratio<100, 1>>>;
#else
	using usage_duration_t = std::chrono::microseconds;
#endif



	using common_duration_t = std::chrono::duration<double, std::chrono::seconds::period>;



	// TODO: filter out duplicates; may result in ambiguous operator= errors. Or use make it explicit which counter uses which type (e.g. std::in_place_index)
	// TODO: make extendable
	using duration_t = std::variant<
		std::monostate, 
		common_duration_t ,// lowest common denominator
	//	std::chrono::duration<float, std::chrono::seconds::period>,
		std::chrono::high_resolution_clock::duration, // for wall time
		usage_duration_t // for user/kernel time
#ifdef ROSETTA_PPM_NVIDIA
		,std::chrono::duration<float, std::chrono::milliseconds::period> // Used by CUDA events
#endif
#ifdef ROSETTA_PLATFORM_NVIDIA
		,std::chrono::duration<uint64_t, std::chrono::nanoseconds::period> // Used by cupti
#endif
	>;

	class IterationMeasurement {
		friend class Iteration;
		template <typename I>
		friend class Iterator;
		friend class State;
		friend struct :: Rosetta;
		friend class Scope;
		friend class ::BenchmarkRun;

	public:
	private:
		// TODO: Make extendable (register user measures in addition to predefined ones)
		duration_t values[MeasureCount];
	};
} // namespace rosetta 

#endif /* ROSETTA_COMMON_H */
