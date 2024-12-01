#include <stdio.h>

__global__ void incrementShared() {
    // Declare shared variable for this block
    __shared__ int counter;
    
    // Only first thread initializes the counter
    if (threadIdx.x == 0) {
        counter = 0;
    }
    
    // Make sure counter is initialized before other threads use it
    __syncthreads();
    
    // Each thread increments the counter
    // We need atomicAdd because multiple threads are updating the same variable
    atomicAdd(&counter, 1);
    
    // Wait for all threads to finish incrementing
    __syncthreads();
    
    // First thread prints the final result
    if (threadIdx.x == 0) {
        printf("Block %d final counter: %d\n", blockIdx.x, counter);
    }
}

int main() {
    // Launch 2 blocks with 4 threads each
    incrementShared<<<2, 4>>>();
    cudaDeviceSynchronize();
    
    return 0;
}
