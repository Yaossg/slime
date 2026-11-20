CXX ?= g++
NVCC ?= nvcc
CUDA_ARCH ?= 86
CXXFLAGS ?= -O3 -std=c++17 -pthread -Wall -Wextra -Wpedantic
NVCCFLAGS ?= -O3 -std=c++17
CUDA_FLAGS = -gencode arch=compute_$(CUDA_ARCH),code=sm_$(CUDA_ARCH) -gencode arch=compute_$(CUDA_ARCH),code=compute_$(CUDA_ARCH)
HEADERS := $(wildcard *.hpp *.cuh ../include/slime/*.hpp)
TARGETS := bin/cpu_forward bin/cuda_steps bin/brute bin/reverse bin/reuse

.PHONY: all test-cpu test-gpu clean
all: $(TARGETS)
bin:
	mkdir -p bin
bin/cpu_forward: cpu_forward.cpp common.hpp ../include/slime/solver_core.hpp Makefile | bin
	$(CXX) $(CXXFLAGS) $< -o $@
bin/%: %.cu $(HEADERS) Makefile | bin
	$(NVCC) $(NVCCFLAGS) $(CUDA_FLAGS) $< -o $@
test-cpu: bin/cpu_forward
	python3 test_steps.py
test-gpu: all
	python3 test_steps.py --gpu
	python3 test_reverse.py
clean:
	rm -rf bin
