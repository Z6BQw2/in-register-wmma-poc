#!/bin/bash


# Adapt this for your GPU. This one's for Blackwell Arch.
ARCH_FLAG="-arch=sm_120" 
nvcc poc_latency.cu -o poc ${ARCH_FLAG}

WARMUP_RUNS=${1:-100}
TIMED_RUNS=${2:-100}

echo "--- Démarrage du benchmark robuste ---"
echo "Warm-up: ${WARMUP_RUNS} exécutions..."

for i in $(seq 1 $WARMUP_RUNS)
do
  ./poc > /dev/null
done

echo "Mesure: ${TIMED_RUNS} exécutions..."

{
  for i in $(seq 1 $TIMED_RUNS)
  do
    ./poc
  done
} | awk '
{
  baseline_sum += $1;
  in_register_sum += $2;
}
END {
  baseline_avg = baseline_sum / NR;
  in_register_avg = in_register_sum / NR;
  speedup = baseline_avg / in_register_avg;
  
  printf "\n--- RÉSULTATS FINALS ---\n";
  printf "Latence moyenne Baseline    : %.8f ms\n", baseline_avg;
  printf "Latence moyenne In-Register : %.8f ms\n", in_register_avg;
  printf "Speedup Final               : %.2fx\n", speedup;
}'
