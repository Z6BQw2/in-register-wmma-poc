#!/bin/bash

# Ce script collecte les 3 métriques micro-architecturales clés pour le papier.

EXECUTABLE_SINGLE_GRID="./latency_bench"
EXECUTABLE_LARGE_GRID="./profiling_bench"

echo "==========================================================="
echo "  1. Shared Memory Traffic (Preuve du mécanisme)"
echo "==========================================================="
echo "--- Kernel Baseline ---"
ncu -k Frag_standard_baseline --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum ${EXECUTABLE_SINGLE_GRID}

echo "\n--- Kernel In-Register ---"
ncu -k Frag_swapped --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum ${EXECUTABLE_SINGLE_GRID}


echo "\n\n==========================================================="
echo "  2. Achieved Occupancy (Utilisation du GPU sous charge)"
echo "==========================================================="
echo "--- Kernel Baseline ---"
ncu -k Frag_standard_baseline --section Occupancy ${EXECUTABLE_LARGE_GRID}

echo "\n--- Kernel In-Register ---"
ncu -k Frag_swapped --section Occupancy ${EXECUTABLE_LARGE_GRID}


echo "\n\n==========================================================="
echo "  3. Compute Throughput (Efficacité du calcul sous charge)"
echo "==========================================================="
echo "--- Kernel Baseline ---"
ncu -k Frag_standard_baseline --section SpeedOfLight ${EXECUTABLE_LARGE_GRID}

echo "\n--- Kernel In-Register ---"
ncu -k Frag_swapped --section SpeedOfLight ${EXECUTABLE_LARGE_GRID}

echo "\n\n--- Analyse terminée ---"
