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

# Original two microbenchmarks + the 10 full HeteroBench co-execution drivers.
MICRO = coexec_gemm coexec_stencil
SUITE = coexec_3mm coexec_mlp coexec_oha coexec_cnn coexec_adi \
        coexec_ced coexec_sbf coexec_opf coexec_spf coexec_ppc

all: $(MICRO) $(SUITE)
suite: $(SUITE)

$(MICRO) $(SUITE): %: %.cu coexec_common.h
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(MICRO) $(SUITE)

.PHONY: all suite clean
