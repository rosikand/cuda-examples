#include <stdio.h>

/*
Each thread starts with its own number
In each step, half the threads combine their values with values from other threads
The stride halves each time, creating a tree-like reduction
Finally, thread 0 has the sum of all values
This demonstrates how threads can pass messages to collaborate on a computation, with each thread doing part of the work and passing results to other threads.

Remember:

Shared memory is fast but only works within a block
Global memory works across blocks but is slower
Warp-level operations are fastest but most restricted
Always use __syncthreads() when needed to prevent race conditions
*/


__global__ void collaborativeSum() {
    __shared__ int data[32];
    int tid = threadIdx.x;

    // Each thread generates its own data
    data[tid] = tid + 1;
    __syncthreads();

    // Collaborative sum: each thread adds two numbers
    for (int stride = 16; stride > 0; stride >>= 1) {
        if (tid < stride) {
            data[tid] += data[tid + stride];
            printf("Thread %d combined %d with %d to get %d\n",
                   tid, data[tid] - data[tid + stride],
                   data[tid + stride], data[tid]);
        }
        __syncthreads();
    }

    // Thread 0 has the final sum
    if (tid == 0) {
        printf("Final sum: %d\n", data[0]);
    }
}

int main() {
    collaborativeSum<<<1, 32>>>();
    cudaDeviceSynchronize();
    return 0;
}
