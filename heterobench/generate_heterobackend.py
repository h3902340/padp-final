#!/usr/bin/env python3
"""Generate heterobackend directories with AI-optimized kernel placement."""

import json
import os
import shutil
from pathlib import Path

HB_DIR = Path(__file__).parent.parent / "HeteroBench" / "HeteroBench"
RESULTS_DIR = Path(__file__).parent / "results"
ASSIGNMENTS_FILE = RESULTS_DIR / "optimal_assignments.json"

# Benchmark name mapping
BENCH_DIRS = {
    "ced": "canny_edge_detection",
    "sbf": "sobel_filter",
    "opf": "optical_flow",
    "cnn": "convolutional_neural_network",
    "mlp": "multilayer_perceptron",
    "oha": "one_head_attention",
    "dgr": "digit_recog",
    "spf": "spam_filter",
    "3mm": "3_matrix_multiplication",
    "adi": "alternating_direction_implicit",
    "ppc": "parallelize_particle",
}

# Kernel source file mapping (from proj_config.json krnl_sources)
KERNEL_FILES = {
    "ced": ["gaussian_filter.cpp", "gradient_intensity_direction.cpp",
            "edge_thinning.cpp", "double_thresholding.cpp", "hysteresis.cpp"],
    "sbf": ["sobel_filter_x.cpp", "sobel_filter_y.cpp", "compute_gradient_magnitude.cpp"],
    "opf": ["gradient_xy_calc.cpp", "gradient_z_calc.cpp", "gradient_weight_y.cpp",
            "gradient_weight_x.cpp", "outer_product.cpp", "tensor_weight_y.cpp",
            "tensor_weight_x.cpp", "flow_calc.cpp"],
    "cnn": ["conv2d.cpp", "relu.cpp", "max_pooling.cpp", "pad_input.cpp", "dot_add.cpp", "softmax.cpp"],
    "mlp": ["dot_add.cpp", "sigmoid.cpp", "softmax.cpp"],
    "oha": ["transpose.cpp", "softmax.cpp", "matmul.cpp"],
    "dgr": ["popcount.cpp", "update_knn.cpp", "knn_vote.cpp"],
    "spf": ["dotProduct.cpp", "Sigmoid.cpp", "computeGradient.cpp", "updateParameter.cpp"],
    "3mm": ["kernel_3mm_0.cpp", "kernel_3mm_1.cpp", "kernel_3mm_2.cpp"],
    "adi": ["init_array.cpp", "kernel_adi.cpp"],
    "ppc": ["compute_forces.cpp", "move_particles.cpp"],
}

MAKEFILE_TEMPLATE = """CPU_CXX := g++
CPU_CXXFLAGS := -Wall -Wextra -pedantic -std=c++11 -fopenmp -w
CPU_LDFLAGS := -fopenmp

GPU_CXX := nvc++
GPU_CXXFLAGS := -Wall -Wextra
GPU_LDFLAGS :=

OPENMP_OFFLOAD_LIBS := -mp=gpu

ITERATIONS := {iterations}
{macro_block}
TARGET_EXEC := {target}_sw

SOURCES := {cpu_sources}

GPU_SOURCES := \\
{gpu_sources}

MACRO := {macro_def}

OBJECTS := $(SOURCES:.cpp=.o)
GPU_OBJECTS := $(GPU_SOURCES:.cpp=.o)

INCLUDE_DIRS := cpu_impl/include gpu_impl/include .
INCLUDE_PARAMS := $(addprefix -I, $(INCLUDE_DIRS))

all: $(TARGET_EXEC)

$(TARGET_EXEC): $(OBJECTS) $(GPU_OBJECTS)
\t$(GPU_CXX) $(GPU_LDFLAGS) $(OPENMP_OFFLOAD_LIBS) -o $@ $^

main.o: main.cpp
\t$(CPU_CXX) $(CPU_CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

cpu_impl/%.o: cpu_impl/%.cpp
\t$(CPU_CXX) $(CPU_CXXFLAGS) $(MACRO) $(INCLUDE_PARAMS) -c $< -o $@

gpu_impl/%.o: gpu_impl/%.cpp
\t$(GPU_CXX) $(GPU_CXXFLAGS) $(MACRO) $(OPENMP_OFFLOAD_LIBS) $(INCLUDE_PARAMS) -c $< -o $@

run:
\t./$(TARGET_EXEC) {run_args}

clean:
\trm -f $(TARGET_EXEC) $(OBJECTS) $(GPU_OBJECTS)

.PHONY: all clean run
"""

# Per-benchmark Makefile parameters
BENCH_PARAMS = {
    "ced": {
        "iterations": 20, "target": "ced",
        "macro_block": """LOW_THRESHOLD := 30
HIGH_THRESHOLD := 90
INPUT_PATH := ../../input/1439x1919_stanford.jpg
OUTPUT_PATH := ../../output/1439x1919_stanford.jpg""",
        "macro_def": "-DITERATIONS=$(ITERATIONS)",
        "run_args": "$(INPUT_PATH) $(OUTPUT_PATH) $(LOW_THRESHOLD) $(HIGH_THRESHOLD)",
    },
    "3mm": {
        "iterations": 20, "target": "krnl_3mm",
        "macro_block": """NI := 1024
NJ := 1024
NK := 1024
NL := 1024
NM := 1024""",
        "macro_def": "-DITERATIONS=$(ITERATIONS) -DNI=$(NI) -DNJ=$(NJ) -DNK=$(NK) -DNL=$(NL) -DNM=$(NM)",
        "run_args": "",
    },
    "sbf": {
        "iterations": 20, "target": "sbf",
        "macro_block": """INPUT_PATH := ../../input/1439x1919_stanford.jpg
OUTPUT_PATH := ../../output/1439x1919_stanford.jpg""",
        "macro_def": "-DITERATIONS=$(ITERATIONS)",
        "run_args": "$(INPUT_PATH) $(OUTPUT_PATH)",
    },
}


def setup_heterobackend(bench_abbr, assignment_bits, kernel_files):
    """Create heterobackend/cpu_gpu_ai directory with optimal kernel split."""
    bench_dir = BENCH_DIRS.get(bench_abbr)
    if not bench_dir:
        print(f"Unknown benchmark: {bench_abbr}")
        return None

    base = HB_DIR / "benchmarks" / bench_dir
    hetero_dir = base / "heterobackend" / "cpu_gpu_ai"
    cpu_src = base / "homobackend_cpu" / "OpenMP"
    gpu_src = base / "homobackend_gpu" / "OpenMP"

    if not cpu_src.exists() or not gpu_src.exists():
        print(f"Missing homobackend for {bench_abbr}")
        return None

    # Clean and recreate
    if hetero_dir.exists():
        shutil.rmtree(hetero_dir)
    hetero_dir.mkdir(parents=True)
    (hetero_dir / "cpu_impl").mkdir()
    (hetero_dir / "cpu_impl" / "include").mkdir()
    (hetero_dir / "gpu_impl").mkdir()
    (hetero_dir / "gpu_impl" / "include").mkdir()

    cpu_sources = []
    gpu_sources = []

    for i, kfile in enumerate(kernel_files):
        if assignment_bits[i] == 0:  # CPU
            src = cpu_src / "cpu_impl" / kfile
            dst = hetero_dir / "cpu_impl" / kfile
            cpu_sources.append(f"    cpu_impl/{kfile}")
        else:  # GPU
            src = gpu_src / "gpu_impl" / kfile
            dst = hetero_dir / "gpu_impl" / kfile
            gpu_sources.append(f"    gpu_impl/{kfile}")

        if src.exists():
            shutil.copy2(src, dst)
        else:
            print(f"  Warning: {src} not found")

    # Copy headers
    for hdr in ["cpu_impl.h"]:
        src = cpu_src / "cpu_impl" / "include" / hdr
        if src.exists():
            shutil.copy2(src, hetero_dir / "cpu_impl" / "include" / hdr)
    for hdr in ["gpu_impl.h"]:
        src = gpu_src / "gpu_impl" / "include" / hdr
        if src.exists():
            shutil.copy2(src, hetero_dir / "gpu_impl" / "include" / hdr)

    # Copy main.cpp: prefer existing heterobackend main (includes both headers)
    hetero_ref = base / "heterobackend" / "cpu_gpu" / "main.cpp"
    if hetero_ref.exists():
        shutil.copy2(hetero_ref, hetero_dir / "main.cpp")
    else:
        main_src = cpu_src / "main.cpp"
        if main_src.exists():
            main_text = main_src.read_text()
            if "gpu_impl.h" not in main_text and gpu_sources:
                main_text = main_text.replace('#include "cpu_impl.h"',
                                              '#include "gpu_impl.h"\n#include "cpu_impl.h"')
            (hetero_dir / "main.cpp").write_text(main_text)

    # Copy supporting files from CPU backend
    extra_sources = []
    for fname in ["init_array.cpp", "init_array.h", "stb_image.h", "stb_image_write.h"]:
        src = cpu_src / fname
        if src.exists():
            shutil.copy2(src, hetero_dir / fname)
            if fname.endswith(".cpp") and fname != "main.cpp":
                extra_sources.append(f"    {fname}")

    # Copy imageLib for opf if needed
    if bench_abbr == "opf":
        imagelib = base / "imageLib"
        if imagelib.exists():
            dst_lib = hetero_dir / "imageLib"
            if not dst_lib.exists():
                shutil.copytree(imagelib, dst_lib)

    # Generate Makefile
    params = BENCH_PARAMS.get(bench_abbr, {
        "iterations": 20,
        "target": bench_abbr,
        "macro_block": "",
        "macro_def": "-DITERATIONS=$(ITERATIONS)",
        "run_args": "",
    })

    makefile = MAKEFILE_TEMPLATE.format(
        iterations=params["iterations"],
        target=params["target"],
        macro_block=params["macro_block"],
        macro_def=params["macro_def"],
        cpu_sources=(" \\\n".join(["main.cpp"] + extra_sources + cpu_sources) if (cpu_sources or extra_sources)
                     else "main.cpp"),
        gpu_sources=" \\\n".join(gpu_sources) if gpu_sources else "    # none",
        run_args=params["run_args"],
    )
    (hetero_dir / "Makefile").write_text(makefile)
    print(f"Generated heterobackend for {bench_abbr} at {hetero_dir}")
    return hetero_dir


def main():
    if not ASSIGNMENTS_FILE.exists():
        print("Run ai_optimizer.py first")
        return

    with open(ASSIGNMENTS_FILE) as f:
        assignments = json.load(f)

    generated = []
    for bench, data in assignments.items():
        bits = data.get("assignment_bits", [])
        kfiles = KERNEL_FILES.get(bench, [])
        if len(bits) != len(kfiles):
            print(f"Skipping {bench}: assignment/kernel count mismatch")
            continue
        path = setup_heterobackend(bench, bits, kfiles)
        if path:
            generated.append(bench)

    print(f"\nGenerated heterobackend for: {generated}")
    with open(RESULTS_DIR / "generated_hetero.json", "w") as f:
        json.dump(generated, f, indent=2)


if __name__ == "__main__":
    main()
