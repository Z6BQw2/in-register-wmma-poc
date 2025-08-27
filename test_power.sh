#!/bin/bash

TEST_DURATION=10
LOOP_COUNT=2000000

EXECUTABLE="./benchmark"

rm -f power_baseline.txt power_in_register.txt

echo "Kernel 1 = Baseline, Kernel 2 = In-Register"
echo "Test Kernel 1"

${EXECUTABLE} 1 ${LOOP_COUNT} > /dev/null &
BENCH_PID=$!
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=1000 > power_baseline.txt &
SMI_PID=$!

sleep ${TEST_DURATION}

kill ${BENCH_PID}
kill ${SMI_PID}
kill -9 ${BENCH_PID} 2>/dev/null
kill -9 ${SMI_PID} 2>/dev/null

sleep 2

echo "Test Kernel 2"

${EXECUTABLE} 2 ${LOOP_COUNT} > /dev/null &
BENCH_PID=$!
nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits --loop-ms=1000 > power_in_register.txt &
SMI_PID=$!

sleep ${TEST_DURATION}

kill ${BENCH_PID}
kill ${SMI_PID}
kill -9 ${BENCH_PID} 2>/dev/null
kill -9 ${SMI_PID} 2>/dev/null

echo ""
ls -l power_*.txt
