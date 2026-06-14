# Optimal CPU+GPU Co-execution via a Roofline-Grounded Split Model

**Parallel & Distributed Programming — Final Project**
**Category B: Using an AI Agent to analyze and optimize parallel programs**

This project uses an AI agent to find the *optimal mix* of CPU and GPU for a workload —
not by assigning whole kernels to one device, but by running **CPU and GPU concurrently
on the same kernel**, splitting the data domain by a tunable fraction `f` (fraction of work
sent to the GPU). The agent **derives the optimal split `f*` analytically from a roofline
model** instead of brute-forcing every value, and then validates the prediction empirically.

**Baseline to beat:** plain C++/OpenMP (pure CPU). We also compare against pure GPU.

---

## TL;DR — Key Result

| Workload | Class | Pure CPU | Pure GPU | **Co-exec (best)** | Predicted `f*` | Empirical `f*` |
|---|---|---:|---:|---:|---:|---:|
| 3×3 Stencil (8192²) | **memory-bound** | 72.3 ms | 46.5 ms | **43.9 ms** | 0.609 | 0.60 |
| GEMM (2048³) | compute-bound | 362.2 ms | 8.4 ms | 8.4 ms | 0.977 | 1.00 |

**Takeaways (the agent's finding):**
- **Memory-bound** kernels get a **real co-execution win** (1.65× over CPU, 1.06× over GPU): both
  devices contribute bandwidth, and the analytical balance point predicts the measured optimum.
- **Compute-bound** kernels (GEMM) **collapse to the dominant device** — the GPU is ~40× faster, so
  the model correctly predicts `f*→1` and co-execution gives no benefit. The agent *predicts* this
  from device throughput ratios rather than discovering it by exhaustive search.

The full write-up is in [`report/final_report.pdf`](report/final_report.pdf).

---

## The model (why this is not brute force)

Split work `W` so the GPU does fraction `f`, the CPU does `1−f`, concurrently. The makespan is

```
T(f) = max( (1−f)·W / R_cpu ,  f·W / R_gpu )
```

`T(f)` is minimized when both devices finish together, giving a **closed-form optimum**:

```
f* = R_gpu / (R_cpu + R_gpu)            (GPU fraction)
T* = W / (R_cpu + R_gpu)
speedup_vs_GPU = 1 + R_cpu / R_gpu
```

`R_cpu`, `R_gpu` are throughputs read off the roofline:
- **compute-bound** → attainable GFLOP/s,
- **memory-bound** → attainable GB/s.

The agent measures the two single-device calibration points (`f=0`, `f=1`), plugs them into the
formula, and predicts `f*` in **one shot**. The empirical sweep over `f` is used only to *confirm*
the prediction lands on the minimum.

---

## Repository layout

```
padp-final/
├── README.md                     ← you are here
├── report/
│   ├── final_report.tex          ← IEEE-format final report (source)
│   └── final_report.pdf          ← compiled report
├── coexec/                       ← CORE: concurrent CPU+GPU co-execution
│   ├── coexec_gemm.cu            ← compute-bound micro-benchmark (CUDA + OpenMP)
│   ├── coexec_stencil.cu         ← memory-bound micro-benchmark (CUDA + OpenMP)
│   ├── Makefile
│   ├── run_coexec.sh             ← builds, sweeps f, runs the analyzer
│   ├── analyze_coexec.py         ← roofline f* prediction + validation
│   └── results/                  ← committed reference outputs (CSV + JSON)
├── heterobench/                  ← real-benchmark profiling pipeline (optional)
│   ├── setup_heterobench.sh      ← clones + patches HeteroBench (multi-GB, not committed)
│   ├── parse_profile.py          ← extracts per-kernel timings from logs
│   ├── ai_optimizer.py           ← roofline classification + assignment search
│   ├── generate_heterobackend.py ← emits custom cpu_gpu backends
│   ├── progress.py               ← live SLURM progress monitor
│   └── results/                  ← pure-CPU/GPU profiles used in the report
└── docs/                         ← project proposal + cluster guide (PDF)
```

---

## Requirements

- **NVIDIA GPU** + CUDA toolkit (`nvcc`). Developed on a **Tesla V100 (sm_70)**.
- A host compiler with **OpenMP** (`g++`/`gcc`).
- Python 3 (standard library only for the core experiment; `numpy`/`matplotlib` optional for plots).
- *(Optional)* SLURM, if you run on a cluster like TWCC Taiwania-2.

No external Python packages are required to reproduce the headline result.

---

## Quick start — reproduce the core result (~2 minutes)

On any machine with a GPU + CUDA + OpenMP:

```bash
cd coexec
make                 # builds coexec_gemm and coexec_stencil (override GPU arch: make ARCH=sm_80)
./run_coexec.sh      # sweeps f for both kernels, writes results/, prints the analysis
```

On a **SLURM cluster** (e.g. TWCC), submit it as a job instead:

```bash
cd coexec
sbatch run_coexec.sh   # edit the -A <account> / module loads at the top for your site
```

Expected console output (numbers vary by GPU):

```
===== stencil (memory-bound) =====
  R_cpu = ... GB/s   R_gpu = ... GB/s
  Predicted f* (GPU fraction) = 0.609
  Empirical best f            = 0.600
  Pure CPU : 72.26 ms
  Pure GPU : 46.49 ms
  Co-exec  : 43.85 ms (predicted 41.0 ms)
  Speedup vs CPU = 1.65x   vs GPU = 1.06x
```

Outputs land in `coexec/results/`:
- `gemm_sweep.csv`, `stencil_sweep.csv` — raw `f`-sweep timings,
- `coexec_analysis.json` — predicted `f*`, measured optimum, speedups.

### Run a single point manually

```bash
./coexec_stencil  <H> <W> <f> <reps>     # e.g. ./coexec_stencil 8192 8192 0.6 7
./coexec_gemm     <N> <f> <reps>         # e.g. ./coexec_gemm 2048 0.5 5
# prints CSV: kernel,...,frac,t_total_ms,t_cpu_ms,t_gpu_ms,(gflops|gbps)
```

Tune the sweep with env vars: `STEN_H`, `STEN_W`, `GEMM_N`, `OMP_NUM_THREADS`, `ARCH`.

---

## Optional — reproduce the HeteroBench profiling (heavy, hours)

This regenerates the pure-CPU vs pure-GPU baselines for the real benchmarks in the report.
It downloads the multi-GB HeteroBench suite, so it is **not** committed here.

```bash
cd heterobench
./setup_heterobench.sh        # clones HeteroBench, patches compilers to g++/nvc++ + -O3
# then profile (long-running; submit under SLURM on a GPU node):
python3 parse_profile.py      # parse logs -> per-kernel timings
python3 ai_optimizer.py       # roofline class + CPU/GPU assignment
python3 progress.py           # (optional) live progress bar for running jobs
```

Reference results that the report uses are already provided in
`heterobench/results/heterobench_pure.json` so you do **not** need to run this to read the report.

---

## Rebuilding the report PDF

```bash
cd report
# Any TeX engine works; we used Tectonic:
tectonic final_report.tex
# or:  pdflatex final_report.tex && bibtex final_report && pdflatex final_report.tex (x2)
```

---

## Hardware / environment used

- **Cluster:** TWCC Taiwania-2, 1× NVIDIA **Tesla V100** GPU node.
- **CPU:** 4 cores allocated, `OMP_NUM_THREADS=4`, `-O3 -march=native -funroll-loops`.
- **GPU compiler:** `nvc++` / `nvcc` with `-O3`, arch `sm_70`.

Results scale to other GPUs — just set `make ARCH=sm_XX` and re-run; the analytical model
re-predicts `f*` from your machine's measured throughputs automatically.
