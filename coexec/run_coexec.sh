#!/bin/bash
# Portable driver for the concurrent CPU+GPU co-execution sweeps.
#
# Run directly on a GPU node, OR submit with sbatch on a SLURM cluster:
#   sbatch run_coexec.sh             (TWCC: edit -A account below)
#   ./run_coexec.sh                  (interactive GPU node)
#
#SBATCH -J coexec
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH -t 00:40:00
#SBATCH -o coexec_%j.out
#SBATCH -e coexec_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# --- Optional: load modules on TWCC Taiwania-2 (no-op elsewhere) ---
if command -v module >/dev/null 2>&1; then
  module load cuda/12.3 2>/dev/null || true
  module load nvhpc-24.11_hpcx-2.20_cuda-12.6 2>/dev/null || true
  module load gcc9/9.3.1 2>/dev/null || true
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ARCH="${ARCH:-sm_70}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== OMP_NUM_THREADS=$OMP_NUM_THREADS  ARCH=$ARCH ==="

echo "=== Compiling ==="
make ARCH="$ARCH"

mkdir -p results
GEMM_CSV=results/gemm_sweep.csv
STEN_CSV=results/stencil_sweep.csv
echo "kernel,N,frac,t_total_ms,t_cpu_ms,t_gpu_ms,gflops" > "$GEMM_CSV"
echo "kernel,H,W,frac,t_total_ms,t_cpu_ms,t_gpu_ms,gbps" > "$STEN_CSV"

GN="${GEMM_N:-2048}"
echo "=== GEMM sweep (compute-bound) N=$GN ==="
for f in 0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.70 0.75 0.80 0.85 0.90 0.95 1.00; do
  ./coexec_gemm "$GN" "$f" 5 | tee -a "$GEMM_CSV"
done

SH="${STEN_H:-8192}"; SW="${STEN_W:-8192}"
echo "=== Stencil sweep (memory-bound) ${SH}x${SW} ==="
for f in 0.00 0.10 0.20 0.30 0.40 0.50 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.00; do
  ./coexec_stencil "$SH" "$SW" "$f" 7 | tee -a "$STEN_CSV"
done

echo "=== Analyzing (roofline f* vs measured) ==="
python3 analyze_coexec.py || true
echo "=== Done. See results/coexec_analysis.json ==="
