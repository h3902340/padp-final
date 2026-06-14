#!/bin/bash
# Build and sweep the split fraction f for all 10 HeteroBench co-execution
# drivers, collecting pure-CPU (f=0), pure-GPU (f=1) and mix (best f) times.
#
# Submit on a SLURM cluster (TWCC):  sbatch run_suite.sh
# Or run directly on an interactive GPU node:  ./run_suite.sh
#
#SBATCH -J coexec_suite
#SBATCH -A ACD115083
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH -c 4
#SBATCH -t 00:40:00
#SBATCH -o suite_%j.out
#SBATCH -e suite_%j.err

set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

if command -v module >/dev/null 2>&1; then
  module load cuda/12.6 2>/dev/null || true
  module load nvhpc-24.11_hpcx-2.20_cuda-12.6 2>/dev/null || true
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export OMP_PLACES=cores
export OMP_PROC_BIND=close
ARCH="${ARCH:-sm_70}"

echo "=== Node: $(hostname) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true
echo "=== Build ==="
make suite ARCH="$ARCH" -j 4 || exit 1

mkdir -p results
CSV=results/suite_sweep.csv
echo "name,frac,t_total_ms,t_cpu_ms,t_gpu_ms,checksum" > "$CSV"

FRACS="0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0"
# reps per benchmark (slow ones get fewer)
declare -A REPS=( [3mm]=3 [mlp]=1 [oha]=3 [cnn]=2 [adi]=2 [ced]=3 [sbf]=3 [opf]=2 [spf]=2 [ppc]=2 )

for b in 3mm mlp oha cnn adi ced sbf opf spf ppc; do
  echo "=== sweep $b (reps=${REPS[$b]}) ==="
  for fr in $FRACS; do
    timeout 300 ./coexec_$b "$fr" "${REPS[$b]}" >> "$CSV" || echo "$b,$fr,TIMEOUT/FAIL"
  done
done

echo "=== Analyze ==="
python3 analyze_suite.py || true
echo "=== Done: $CSV and results/suite_results.json ==="
