NVCC = nvcc

ARCH = -arch=sm_120
CFLAGS = $(ARCH) -O3 -I. # -I. pour inclure les headers du dossier courant

TARGET_CORRECTNESS = correctness
TARGET_LATENCY = latency_bench
TARGET_POWER = power_bench
TARGET_PROFILING = profiling_bench

TARGETS = $(TARGET_CORRECTNESS) $(TARGET_LATENCY) $(TARGET_POWER) $(TARGET_PROFILING)

SRC_CORRECTNESS = main_correctness.cu
SRC_LATENCY = main_runtime.cu
SRC_POWER = main_power.cu
SRC_PROFILING = main_profiling.cu

DEPS = kernels.cuh utils.cuh

all: $(TARGETS)

$(TARGET_CORRECTNESS): $(SRC_CORRECTNESS) $(DEPS)
	$(NVCC) $(CFLAGS) $< -o $@

$(TARGET_LATENCY): $(SRC_LATENCY) $(DEPS)
	chmod +x benchmark_latency.sh
	$(NVCC) $(CFLAGS) $< -o $@

$(TARGET_POWER): $(SRC_POWER) $(DEPS)
	chmod +x benchmark_power.sh
	$(NVCC) $(CFLAGS) $< -o $@

$(TARGET_PROFILING): $(SRC_PROFILING) $(DEPS)
	chmod +x benchmark_profiling.sh
	$(NVCC) $(CFLAGS) $< -o $@

test: all run_correctness run_latency run_power run_profiling

run_correctness: $(TARGET_CORRECTNESS)
	@echo "\n--- Correctness Test (if this test is incorrect, verify your choice of architecture within the makefile) ---"
	./$(TARGET_CORRECTNESS)

run_latency: $(TARGET_LATENCY)
	@echo "\n--- Runtime Benchmark ---"
	./benchmark_latency.sh

run_power: $(TARGET_POWER)
	@echo "\n--- Power Benchmark ---"
	./benchmark_power.sh

run_profiling: $(TARGET_PROFILING)
	@echo "\n--- NCU profiling ---"
	./benchmark_profiling.sh

clean:
	@echo "Cleaning files"
	rm -f $(TARGETS) *.o *.ncu-rep *.tmp

.PHONY: all test run_correctness run_latency run_power run_profiling clean
