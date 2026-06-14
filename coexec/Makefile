# Concurrent CPU+GPU co-execution micro-benchmarks
# Requires: CUDA toolkit (nvcc) + an OpenMP-capable host compiler.
#
# Build:  make
# Clean:  make clean
#
# ARCH defaults to sm_70 (Tesla V100). Override for your GPU, e.g.:
#   make ARCH=sm_80   # A100
#   make ARCH=sm_86   # RTX 30xx
#   make ARCH=sm_90   # H100

ARCH ?= sm_70
NVCC ?= nvcc
NVCCFLAGS = -O3 -arch=$(ARCH) -Xcompiler "-O3 -march=native -fopenmp"

all: coexec_gemm coexec_stencil

coexec_gemm: coexec_gemm.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

coexec_stencil: coexec_stencil.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f coexec_gemm coexec_stencil

.PHONY: all clean
