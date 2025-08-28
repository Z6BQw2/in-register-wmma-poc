For every benchmark, replace ARCH_FLAG with your own architecture; otherwise, the outputs will be nonsensical.
You can ensure that your architecture choice was correct by running poc.cu and verifying that your output is correct.

bench_latency.sh outputs the speed of both kernels in their simplest form, looped over in the shell script to avoid launch overhead and dead code.
You can choose the number of warm-up passes and averaged-out passes by passing them as arguments: sh bench_latency.sh <int: warm-ups> <int: passes>

bench_power.sh outputs the average power consumption measured by nvidia-smi during $TEST_DURATION seconds during the looped execution of the kernels.
You can choose the duration of the warm-ups and of the kernel loop by passing them as arguments: sh bench_power.sh <int: warm-up-length> <int:loop-length>
You can manually change the number of measurements per unit of time by manipulating loop-ms, but given the stability of the results, that is not very useful.

bench_general.sh outputs raw data of many metrics regarding the execution of the kernels (occupancy, IPC, etc).

FlashAttentionInRegister.cu provides an example of a working implementation of in-frag manipulation to a Flash Attention kernel. That version is pretty simple (no double buffering for V, no warp specialization, etc.), so implementing the method for more complex kernels might, however, prove more difficult.
