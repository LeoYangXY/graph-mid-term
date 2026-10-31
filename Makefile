# H20/H100: sm_90; A100: sm_80; RTX 5050 Laptop: sm_120
ARCH ?= sm_90
KHOP ?= 5
SPH ?= 128
CXX ?= g++
NVCC ?= nvcc
BUILD = build
INDEX = $(BUILD)/iex-k$(KHOP)-p$(SPH)

all: $(BUILD)/paper $(BUILD)/qgpu $(INDEX)

$(BUILD)/paper: src/paper.cpp
	mkdir -p $(BUILD)
	$(CXX) -O2 -o $@ $<

$(BUILD)/qgpu: src/q.cu
	mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

$(INDEX): src/iex.cu
	mkdir -p $(BUILD)
	$(NVCC) -O3 -arch=$(ARCH) -DKHOP=$(KHOP) -DSPH=$(SPH) -o $@ $<

test: all
	python3 scripts/test_correctness.py --builder $(INDEX) --reader $(BUILD)/paper

clean:
	rm -rf $(BUILD)

.PHONY: all test clean
