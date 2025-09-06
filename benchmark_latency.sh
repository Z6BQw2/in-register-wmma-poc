#!/bin/bash

EXECUTABLE="./latency_bench"

WARMUP_RUNS=${1:-100}
TIMED_RUNS=${2:-100}

echo "--- Démarrage du benchmark robuste ---"
echo "Warm-up: ${WARMUP_RUNS} exécutions..."

for i in $(seq 1 $WARMUP_RUNS)
do
  ${EXECUTABLE} > /dev/null
done

echo "Mesure: ${TIMED_RUNS} exécutions..."

{
  for i in $(seq 1 $TIMED_RUNS)
  do
    ${EXECUTABLE}
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
