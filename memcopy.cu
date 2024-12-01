#include <stdio.h>

__global__ void multiplyByTwo(int *data) {
    int idx = threadIdx.x;
    if (data[idx] % 2 == 0) {
        data[idx] *= 2;
    } else {
        data[idx] *= 1;
    }
}

int main() {
    // CPU data that we want to process
    int h_input[4] = {1, 2, 3, 4};  // This is in CPU memory
    int h_output[4];                 // For storing results

    // Allocate GPU memory
    int *d_data;
    cudaMalloc(&d_data, 4 * sizeof(int));

    // NEED #1: Copy input to GPU to process it
    cudaMemcpy(d_data, h_input, 4 * sizeof(int), cudaMemcpyHostToDevice);

    // Process on GPU
    multiplyByTwo<<<1, 4>>>(d_data);

    // NEED #2: Copy results back to CPU to use them
    cudaMemcpy(h_output, d_data, 4 * sizeof(int), cudaMemcpyDeviceToHost);

    // Now we can use the results on CPU
    printf("Results: %d, %d, %d, %d\n",
           h_output[0], h_output[1], h_output[2], h_output[3]);

    cudaFree(d_data);
    return 0;
}
