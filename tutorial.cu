/* File: tutorial.cu
 * ------------
 * This file contains implementations and invokations of several CUDA examples
 * where each one showcases a new concept. 
 */


#include <stdio.h>
#include <cstdlib>
#include <iostream>


// Kernel function to print "Hello, World!" from each thread
__global__ void helloWorldKernel() {
    printf("Hello, World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}



// Kernel demonstrating simple parameter passing
__global__ void simpleParamsKernel(int a, float b, bool c, char d) {
    // Each thread prints its ID and the received values
    printf("Thread [%d,%d]:\n"
           "    Received int: %d\n"
           "    Received float: %.2f\n"
           "    Received bool: %s\n"
           "    Received char: %c\n",
           blockIdx.x, threadIdx.x,
           a, b,
           c ? "true" : "false", d);
}



// Kernel showing arithmetic with passed parameters
__global__ void calculationKernel(int x, int y) {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;


    // Print the global thread index and full blockIdx
    printf("Global Thread Index: %d | blockIdx: (%d, %d, %d) | blockDim: (%d, %d, %d) | threadIdx: (%d, %d, %d)\n",
           threadId,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z);


    // Each thread performs different operations with the same input values
    int sum = x + y;
    int product = x * y;
    int threadSpecific = sum + threadId;
    
    printf("Thread %d: x=%d, y=%d\n"
           "    Sum: %d\n"
           "    Product: %d\n"
           "    Thread-specific result: %d\n",
           threadId, x, y, sum, product, threadSpecific);
}



__global__ void printKernel() {
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Print the global thread index and full blockIdx
    printf("Global Thread Index: %d | blockIdx: (%d, %d, %d) | blockDim: (%d, %d, %d) | threadIdx: (%d, %d, %d)\n",
           globalIndex,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}


__global__ void printKernel3D() {
    // Calculate global index for 3D configuration
    int globalIndex = (blockIdx.z * gridDim.y * gridDim.x + 
                      blockIdx.y * gridDim.x + 
                      blockIdx.x) * (blockDim.x * blockDim.y * blockDim.z) +
                      (threadIdx.z * blockDim.y * blockDim.x +
                       threadIdx.y * blockDim.x +
                       threadIdx.x);
    
    // Print the global thread index and full blockIdx
    printf("Global Thread Index: %d | blockIdx: (%d, %d, %d) | blockDim: (%d, %d, %d) | threadIdx: (%d, %d, %d)\n",
           globalIndex,
           blockIdx.x, blockIdx.y, blockIdx.z,
           blockDim.x, blockDim.y, blockDim.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}


// Example showing different parameter types
__global__ void parameterDemo(
    int simpleValue,        // Passed by value
    int* arrayValue        // Requires cudaMalloc
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    printf("Thread %d:\n"
           "  simpleValue: %d (passed directly)\n"
           "  arrayValue[%d]: %d (needed cudaMalloc)\n",
           idx, simpleValue, idx, arrayValue[idx]);

    __syncthreads();
    
    // now modify 
    printf("\n Modifying...\n");
    arrayValue[idx] = arrayValue[idx] * 11;
    printf("Thread %d:\n"
           "  arrayValue[%d]: %d (modified)\n",
           idx, idx, arrayValue[idx]);

}


// Simple *correc* synchronization example
__global__ void correctSync(int *data) {
    int tid = threadIdx.x;  // using 1D blocking 

    // print current threadIdx 
    printf("Thread %d\n", tid);
    
    // First half of threads write
    if (tid < 3) {
        data[tid] = (tid + 1) * 1;
    }
    
    // Wait for all writes to complete
    __syncthreads();
    
    // Second half of threads read and modify
    if (tid >= 3) {
        // Read value from corresponding thread in first half and multiply it by 11 
        data[tid] = data[tid - 3] * 11;
    }
}




int main() {
    // -----------------------------------------------------------------------------------
    // ********* Example 0: Hello, World! Kernel ********
    // Description: Prints "Hello, World!" from each thread
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 0: Hello, World! Kernel ********\n";
    std::cout << "// Description: Prints \"Hello, World!\" from each thread\n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";
    

    // Launch the kernel with 2 block of 4 threads
    helloWorldKernel<<<2, 4>>>();

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 1: Passing multiple parameters of different types ********
    // Description: Demonstrates passing multiple parameters of different types to a kernel 
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 1: Passing multiple parameters of different types ********\n";
    std::cout << "// Description: Demonstrates passing multiple parameters of different types to a kernel\n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";

    int a = 42;
    float b = 3.14f;
    bool c = true;
    char d = 'X';
    
    printf("Launching kernel with parameters:\n"
           "a=%d, b=%.2f, c=%s, d=%c\n\n",
           a, b, c ? "true" : "false", d);



           
    // Launch with 2 blocks, 3 threads each
    simpleParamsKernel<<<2, 3>>>(a, b, c, d);
    cudaDeviceSynchronize();



    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 2: Showing how threads can use the same input differently ********
    // Description: Shows how threads can use the same input differently 
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 2: Showing how threads can use the same input differently ********\n";
    std::cout << "// Description: Shows how threads can use the same input differently \n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";
    
    int x = 10;
    int y = 5;
    
    printf("Launching calculation kernel with x=%d, y=%d\n\n", x, y);
    
    // Launch with 1 block, 4 threads
    calculationKernel<<<1, 4>>>(x, y);
    cudaDeviceSynchronize();


    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 3: just a quick example showing thread indexing ********
    // Description: showcases thread indexing in the 2d (blockNum, threadNum) case 
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 3: just a quick example showing thread indexing ********\n";
    std::cout << "// Description: showcases thread indexing in the 2d (blockNum, threadNum) case \n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";

    // Launch with num_blocks blocks, num_threads threads
    int num_blocks = 4;
    int num_threads = 4;
    printf("Launching print kernel with %d blocks and %d threads\n\n", num_blocks, num_threads);
    printKernel<<<num_blocks, num_threads>>>();
    cudaDeviceSynchronize();


    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 4: 3D grid/block kernel example ********
    // Description: showcases same as above but with (grid, block) dimensions in 3D for each instead 
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 4: 3D grid/block kernel example ********\n";
    std::cout << "// Description: showcases same as above but with (grid, block) dimensions in 3D for each instead \n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";
    
    // Define 3D block dimensions
    dim3 blockDim(2, 2, 2);  // 2x2x2 threads per block = 8 threads total per block
    dim3 gridDim(4, 4, 4);   // 2x2x1 blocks = 4 blocks total

    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    int totalThreads = threadsPerBlock * gridDim.x * gridDim.y * gridDim.z;

    std::cout << "Total threads: " << totalThreads << std::endl;

    // Launch kernel
    printKernel3D<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();

    printf("\n----------------------------------------\n\n");


    // -----------------------------------------------------------------------------------
    // ********* Example 5: Passing arrays and copying data via dynamic memory ********
    // Note that this is actually parallelized code! Notice the print statements in the
    // kernel are actually out of order! 
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 5: Passing arrays and copying data via dynamic memory ********\n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";

    // Simple value - no special handling needed
    int xx = 42;
    
    // Array - needs cudaMalloc
    int h_array[5] = {1, 2, 3, 4, 5};  // Host array
    int* d_array;                       // Device array pointer
    cudaMalloc(&d_array, 5 * sizeof(int));
    cudaMemcpy(d_array, h_array, 5 * sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch kernel
    parameterDemo<<<1, 5>>>(xx, d_array);
    cudaDeviceSynchronize();

    // receive results
    printf("\n Received array values:\n");
    cudaMemcpy(h_array, d_array, 5 * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < 5; i++) {  
        std::cout << "h_array[" << i << "] = " << h_array[i] << std::endl;
    }

    // Cleanup
    cudaFree(d_array);  // Make sure to free the dyanamic memory! 


    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 6: A simple synchronization example ********
    // we will explore how to use the previous thread to update the current thread 
    // in this example, we 
    // 1. First half of threads write their thread ID to global memory
    // 2. __syncthreads() ensures all writes complete
    // 3. Second half of threads read the value written by their corresponding thread and add 1 to it
    // We will also show a failure case (called a race condition)
    // -----------------------------------------------------------------------------------

    std::cout << "// -----------------------------------------------------------------------------------\n";
    std::cout << "// ********* Example 6: A simple synchronization example ********\n";
    std::cout << "// -----------------------------------------------------------------------------------\n\n";

    int h_data[6] = {1,2,3,4,5,6};
    int *d_data;  // pointer we will use to copy over host data to device 
    
    cudaMalloc(&d_data, 6 * sizeof(int));  // sets aside n bytes of memory on CUDA device, returns pointer to that memory (pointer itself is stored on host)
    cudaMemset(d_data, 0, 6 * sizeof(int));

    // note: we don't actually need to cudaMemcpyHostToDevice since we haven't edited any memory data on host yet 
    // just pass in the pointer reference to global memory 

    // Test correct synchronization
    printf("Testing correct synchronization:\n");
    correctSync<<<1, 6>>>(d_data);
    cudaMemcpy(h_data, d_data, 6 * sizeof(int), cudaMemcpyDeviceToHost);
    
    // Print some results from the correct version
    for (int i = 0; i < 6; i++) {
        if (i < 3) {
            printf("data[%d] = %d (should be %d)\n", i, h_data[i], (i+1));
        } else {
            printf("data[%d] = %d (should be %d)\n", i, h_data[i], (i-2) * 11);
        }
    }
    
    // Reset data
    cudaMemset(d_data, 0, 6 * sizeof(int));
    
    
    printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 7: A vector summation example ********
    // The hello, world of CUDA. We'll see how to sum a vector in parallel.  
    // -----------------------------------------------------------------------------------

    // printf("\n----------------------------------------\n\n");

    // -----------------------------------------------------------------------------------
    // ********* Example 8: What type of speedup do we get from the parallel vector summation? ********
    // We compare and contrast a sequential and parallel vector summation in terms of speed. 
    // -----------------------------------------------------------------------------------


    // printf("\n----------------------------------------\n\n");
    
    // -----------------------------------------------------------------------------------
    // ********* Example 9: Shared and global memory ********
    // GPU memory is accessible on a per-thread (local, stack), per-block (shared), and per-device (global) basis.
    // We'll explore each one. 
    // -----------------------------------------------------------------------------------



    
    // printf("\n----------------------------------------\n\n");
    return 0;
}
