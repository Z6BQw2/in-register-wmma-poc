#!/bin/bash

# Configuration
EXECUTABLE="./power_bench"
TEST_DURATION=10
WARMUP_DURATION=2

# Fichiers temporaires
BASELINE_FILE="power_baseline.tmp"
IN_REGISTER_FILE="power_in_register.tmp"

# --- Fonction de Nettoyage ---
# Cette fonction sera appelée à la fin du script ou en cas d'interruption
cleanup() {
    # echo "Nettoyage des processus en arrière-plan..."
    # 'pkill -P $$' tue tous les processus enfants du script actuel
    pkill -P $$ > /dev/null 2>&1
    rm -f ${BASELINE_FILE} ${IN_REGISTER_FILE}
}
trap cleanup EXIT INT TERM # Appelle cleanup à la sortie, Ctrl+C, etc.


# --- Fonction de Test ---
test_kernel() {
    local kernel_id=$1
    local kernel_name=$2
    local output_file=$3

    echo ""
    echo "--- TEST PUISSANCE KERNEL ${kernel_id} (${kernel_name}) ---"
    
    # Lance le kernel en boucle en arrière-plan
    while true; do ${EXECUTABLE} ${kernel_id} > /dev/null; done &
    local BENCH_PID=$!
    
    # Laisse le temps au GPU de monter en charge
    sleep ${WARMUP_DURATION}
    
    # Lance nvidia-smi pour logger la puissance
    nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=1000 > ${output_file} &
    local SMI_PID=$!
    
    # Attend la fin de la mesure
    sleep ${TEST_DURATION}
    
    # Arrête les processus. Le 'trap' s'occupera du nettoyage final.
    kill ${SMI_PID} ${BENCH_PID} > /dev/null 2>&1
    wait ${BENCH_PID} > /dev/null 2>&1

    echo "Mesures brutes pour ${kernel_name} sauvegardées."
}

# --- Exécution ---
echo "--- Démarrage du test de puissance ---"

# Teste les deux kernels
test_kernel 1 "Baseline" ${BASELINE_FILE}
test_kernel 2 "In-Register" ${IN_REGISTER_FILE}

# --- Analyse des Résultats ---
analyze_file() {
    local file=$1
    local name=$2
    
    echo ""
    echo "--- Analyse pour ${name} ---"
    
    awk '
    {
        sum += $1;
        count++;
    }
    END {
        if (count > 0) {
            mean = sum / count;
            printf "  - Nombre de mesures : %d\n", count;
            printf "  - Puissance moyenne : %.2f W\n", mean;
        } else {
            print "Erreur: Fichier vide ou illisible.";
        }
    }' ${file}
}

analyze_file ${BASELINE_FILE} "Baseline"
analyze_file ${IN_REGISTER_FILE} "In-Register"

echo ""
echo "--- Tests terminés. ---"
# Le trap s'occupera de supprimer les fichiers .tmp
