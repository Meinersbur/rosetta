#include "rosetta.h"
#include <vector>
#include <cstdio>

// Pointer to ensure no static ctor call avoiding static initialization order fiasco.
static RosettaBenchmark *benchmarkListFirst=nullptr;
static RosettaBenchmark *benchmarkListLast=nullptr;

RosettaBenchmark:: RosettaBenchmark(const char *name, BenchmarkFuncTy &func) : name(name), func(&func) , next(nullptr){
     if (!benchmarkListFirst)
      benchmarkListFirst  = this;
      if (benchmarkListLast) {
          benchmarkListLast->next = this;
          benchmarkListLast = this;
      }
 }


int main(int argc, char* argv[]) {
    ::benchmark::Initialize(&argc, argv);

    int n = 100; // N
    if (argc > 1) {
       n = std::atoi(argv[1]);
       argc -= 1;
       argv += 1;
    }

 printf("main()\n");
    RosettaBenchmark *cur = benchmarkListFirst;
    while (cur) {
        std::string&& name = std::string(cur->getName()) + "/"+std:: to_string(n);
        printf("name %s\n", name.c_str());
        benchmark::RegisterBenchmark(   name.c_str(), cur->getFunc(), n)->Unit(benchmark::kMillisecond);
        cur = cur->getNext();
    }

    //benchmark::RegisterBenchmark(("apirwise.serial" + std::string("/") +std:: to_string(n)).c_str(), pairwise, n)->Unit(benchmark::kMillisecond);
    //benchmark::RegisterBenchmark(("pairwise.cuda" + std::string("/") +std:: to_string(n) ).c_str(), &pairwise_serial, n)->MeasureProcessCPUTime()->UseRealTime()->Unit(benchmark::kMillisecond)->UseManualTime();
    
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return EXIT_SUCCESS;
}
