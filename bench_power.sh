#!/bin/bash

# Adapt this for your GPU. This one's for Blackwell Arch.
ARCH_FLAG="-arch=sm_120" 
nvcc poc_power.cu -o poc ${ARCH_FLAG}


TEST_DURATION=${1:-10}
WARMUP_DURATION=${1:-2}
EXECUTABLE="./poc"

BASELINE_FILE="power_baseline.tmp"
IN_REGISTER_FILE="power_in_register.tmp"


echo "--- Démarrage du test de puissance (Méthode Fichiers Temporaires) ---"

test_kernel() {
    local kernel_id=$1
    local kernel_name=$2
    local output_file=$3

    echo ""
    echo "--- TEST PUISSANCE KERNEL ${kernel_id} (${kernel_name}) ---"
    
    while true; do ${EXECUTABLE} ${kernel_id} > /dev/null; done &
    BENCH_PID=$!
    
    sleep ${WARMUP_DURATION}
    
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=1000 > ${output_file} &
    SMI_PID=$!
    
    sleep ${TEST_DURATION}
    
    kill -9 ${BENCH_PID} ${SMI_PID} 2>/dev/null
    echo "Mesures brutes pour ${kernel_name} sauvegardées."
}

test_kernel 1 "Baseline" ${BASELINE_FILE}
test_kernel 2 "In-Register" ${IN_REGISTER_FILE}

analyze_file() {
    local file=$1
    local name=$2
    
    echo ""
    echo "--- Analyse pour ${name} ---"
    
    awk '
    {
        sum += $1;
        sum_sq += $1 * $1;
        count++;
    }
    END {
        if (count > 0) {
            mean = sum / count;
            variance = (sum_sq / count) - (mean * mean);
            printf "  - Nombre de mesures : %d\n", count;
            printf "  - Puissance moyenne : %.2f W\n", mean;
            printf "  - Variance          : %.2f W^2\n", variance;
        } else {
            print "Erreur: Fichier vide ou illisible.";
        }
    }' ${file}
}

analyze_file ${BASELINE_FILE} "Baseline"
analyze_file ${IN_REGISTER_FILE} "In-Register"

rm -f ${BASELINE_FILE} ${IN_REGISTER_FILE}

echo ""
echo "--- Tests et analyse terminés. Fichiers temporaires supprimés. ---"
