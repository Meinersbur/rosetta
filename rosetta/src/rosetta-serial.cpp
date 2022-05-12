#include "rosetta.h"
#include <vector>
#include <cstdio>

// Pointer to ensure no static ctor call avoiding static initialization order fiasco.
static RosettaBenchmark *benchmarkListFirst=nullptr;
static RosettaBenchmark *benchmarkListLast=nullptr;

RosettaBenchmark::RosettaBenchmark(const char *name, BenchmarkFuncTy &func) : name(name), func(&func) , next(nullptr){
     if (!benchmarkListFirst)
        benchmarkListFirst  = this;
      if (benchmarkListLast) {
          benchmarkListLast->next = this;
          benchmarkListLast = this;
      }
 }

#ifdef _MSC_VER
//__declspec(selectany)
#else
__attribute__((weak))
#endif
void run(benchmark::State& state, int n);

int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    // TODO: benchnark-specific default size
    int n = 100;
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

fprintf(stderr,"main()\n");
    RosettaBenchmark *cur = benchmarkListFirst;
    while (cur) {
        std::string&& name = std::string(cur->getName()) + "/"+std:: to_string(n);
        printf("name %s\n", name.c_str());
        benchmark::RegisterBenchmark(name.c_str(), cur->getFunc(), n)->Unit(benchmark::kMillisecond);
        cur = cur->getNext();
    }

    //benchmark::RegisterBenchmark(("apirwise.serial" + std::string("/") +std:: to_string(n)).c_str(), pairwise, n)->Unit(benchmark::kMillisecond);
    //benchmark::RegisterBenchmark(("pairwise.cuda" + std::string("/") +std:: to_string(n) ).c_str(), &pairwise_serial, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond)->UseManualTime();
    if (&run) {
        BenchmarkFuncTy *func = &run;
        benchmark::RegisterBenchmark(argv[0], func, n)->Unit(benchmark::kMillisecond)->ComputeStatistics("count", [](const std::vector<double>& v) -> double {
            return  std::end(v) - std::begin(v);
            }, benchmark::StatisticUnit::kTime);
    }



    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}
