# GPU arch:  H20 / A100 / H100 -> sm_90     RTX 5050 laptop -> sm_120
ARCH ?= sm_90
PP   ?= 12
CXX  ?= g++
NVCC ?= nvcc
BUILD = build

all: $(BUILD)/paper $(BUILD)/qgpu $(BUILD)/ifrt$(PP)

$(BUILD)/paper: src/paper.cpp
	mkdir -p $(BUILD)
	$(CXX) -O2 -o $@ src/paper.cpp

$(BUILD)/qgpu: src/q.cu
	mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(ARCH) -o $@ src/q.cu

$(BUILD)/ifrt$(PP): src/ifront.cu
	mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(ARCH) -DPP=$(PP) -o $@ src/ifront.cu

clean:
	rm -rf $(BUILD)

.PHONY: all clean
