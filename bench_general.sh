# Adapt this for your GPU. This one's for Blackwell Arch.
ARCH_FLAG="-arch=sm_120" 
nvcc poc_latency.cu -o poc ${ARCH_FLAG}
nvcc poc_profiling.cu -o poc_profiled ${ARCH_FLAG}

echo "Kernel Baseline:"
ncu -k Frag_standard_baseline --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum ./poc

echo "Kernel In-Register:"
ncu -k Frag_swapped --metrics smsp__sass_inst_executed_op_shared_ld.sum,smsp__sass_inst_executed_op_shared_st.sum ./poc

echo ""

echo "Kernel Baseline:"
ncu -k Frag_standard_baseline --section Occupancy ./poc_profiled

echo "Kernel In-Register:"
ncu -k Frag_swapped --section Occupancy ./poc_profiled

echo ""

echo "Kernel Baseline:"
ncu -k Frag_standard_baseline --section SpeedOfLight ./poc_profiled

echo "Kernel In-Register:"
ncu -k Frag_swapped --section SpeedOfLight ./poc_profiled

echo ""

echo "Kernel Baseline:"
ncu -k Frag_standard_baseline --metrics smsp__inst_executed.avg.per_cycle_active ./poc

echo "Kernel In-Register:"
ncu -k Frag_swapped --metrics smsp__inst_executed.avg.per_cycle_active ./poc

echo ""
echo "--- Analyse terminée ---"
