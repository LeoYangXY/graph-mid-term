# GPU arch:  H20/A100/H100 -> sm_90    RTX 5050 laptop -> sm_120
# usage:  make ARCH=sm_90
ARCH ?= sm_90
CXX  ?= g++
NVCC ?= nvcc
CXXFLAGS ?= -O3 -fopenmp
NVFLAGS  ?= -O3 -arch=$(ARCH)
BUILD = build

all: $(BUILD)/tb $(BUILD)/batch_gpu

$(BUILD)/tb: src/tb.cpp src/tb.h
	mkdir -p $(BUILD)
	$(CXX) $(CXXFLAGS) -o $@ src/tb.cpp

$(BUILD)/batch_gpu: src/batch.cu
	mkdir -p $(BUILD)
	$(NVCC) $(NVFLAGS) -o $@ src/batch.cu

clean:
	rm -rf $(BUILD)

.PHONY: all clean
