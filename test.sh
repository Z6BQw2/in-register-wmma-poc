#!/bin/bash

# Adapt this for your GPU. This one's for Hopper Arch.
ARCH_FLAG="-arch=sm_90a" 
nvcc poc.cu -o benchmark ${ARCH_FLAG}
echo "Compilation terminée."

./benchmark > latency_results.txt
echo "Results saved in latency_results.txt"
echo "Profiling Kernel 1 (Baseline)"
ncu --set full --kernel-name Frag_standard_baseline -o profile_baseline.ncu-rep ./benchmark
echo "Profil of Kernel 1 saved in profile_baseline.ncu-rep"

echo "Profiling Kernel 2 (In-register)"
ncu --set full --kernel-name Frag_swapped -o profile_in_register.ncu-rep ./benchmark
echo "Profil of Kernel 2 saved in profile_in_register.ncu-rep"
