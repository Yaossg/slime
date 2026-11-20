NVCC ?= nvcc
CUDA_ARCH ?= 86
NVCCFLAGS ?= -O3 -std=c++17
TARGET := bin/slime-bench
HEADERS := $(wildcard src/*.hpp src/*.cuh) ../include/slime/solver_core.hpp ../include/slime/upper_interval.hpp

.PHONY: all clean test test-gpu
all: $(TARGET)
$(TARGET): slime_bench.cu $(HEADERS) Makefile
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH) slime_bench.cu -o $@
bin/test-math: test_math.cu $(HEADERS) Makefile
	mkdir -p bin
	$(NVCC) $(NVCCFLAGS) -Xcompiler -pthread test_math.cu -o $@
test: $(TARGET) bin/test-math
	bin/test-math
	python3 test_cli.py
test-gpu: $(TARGET)
	python3 test_cli.py --gpu
clean:
	rm -f $(TARGET) bin/test-math
