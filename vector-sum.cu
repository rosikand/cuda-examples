#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>
#include <chrono>

// Kernel function to print "Hello, World!" from each thread
__global__ void vectorSumKernel(const int* A, const int* B, int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }

}

// Sequential vector addition function (here for speedup testing purposes)
void vectorSumSequential(const int* A, const int* B, int* C, int n) {
    for(int i = 0; i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

int main() {
    // Define input arrays 
    const int n = 1000000000;

    int* A = new int[n];
    int* B = new int[n];
    int* C_cuda = new int[n];    // For CUDA results
    int* C_seq = new int[n];     // For sequential results

    // Initialize input arrays
    for (int i = 0; i < n; i++) {
        A[i] = i + 1;
        B[i] = (i + 1) * 10;
    }

    // print first five elements as sample
    printf("\nFirst 5 elements of the arrays:\n");
    for (int i = 0; i < 5; i++) {
        printf("Index %d: A=%d, B=%d\n", i, A[i], B[i]);
    }


    // ================== CUDA Implementation ==================

    auto start_cuda = std::chrono::high_resolution_clock::now();

    // send over to gpu 
    int* a_ptr;
    cudaMalloc(&a_ptr, n * sizeof(int));
    cudaMemcpy(a_ptr, A, n * sizeof(int), cudaMemcpyHostToDevice); 

    int* b_ptr;
    cudaMalloc(&b_ptr, n * sizeof(int));
    cudaMemcpy(b_ptr, B, n * sizeof(int), cudaMemcpyHostToDevice);

    int* c_ptr;
    cudaMalloc(&c_ptr, n * sizeof(int));

    // Calculate grid size based on number of elements (note that anything < num elems won't result in speedups)
    int threadsPerBlock = 512;  // Multiple of 32, half of max (1024)
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    printf("Thread Configuration:\n");
    printf("Threads per block: %d\n", threadsPerBlock);
    printf("Number of blocks: %d\n", blocksPerGrid);
    printf("Total threads: %d\n", threadsPerBlock * blocksPerGrid);
    printf("Array size: %d\n\n", n);
    
    // Launch the kernel with 1 block of 10 threads
    auto start_launch = std::chrono::high_resolution_clock::now();
    vectorSumKernel<<<blocksPerGrid, threadsPerBlock>>>(a_ptr, b_ptr, c_ptr, n);
    
    // Wait for the GPU to finish
    cudaDeviceSynchronize();
    auto end_launch = std::chrono::high_resolution_clock::now();

    // Copy result back to host
    cudaMemcpy(C_cuda, c_ptr, n * sizeof(int), cudaMemcpyDeviceToHost);


    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    auto end_cuda = std::chrono::high_resolution_clock::now();
    
    std::chrono::duration<double, std::milli> cuda_ms = end_cuda - start_cuda;
    std::chrono::duration<double, std::milli> launch_ms = end_launch - start_launch;

    // ================== Sequential Implementation ==================
    
    auto start_seq = std::chrono::high_resolution_clock::now();
    
    vectorSumSequential(A, B, C_seq, n);
    
    auto end_seq = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> seq_ms = end_seq - start_seq;


    // ================== Verify Results ==================

    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (C_cuda[i] != C_seq[i]) {
            printf("Mismatch at index %d: CUDA=%d, Sequential=%d\n", i, C_cuda[i], C_seq[i]);
            correct = false;
            break;
        }
    }

    printf("------------------------------------------\n");
    printf("\nResults for vector addition of %d elements:\n", n);
    printf("CUDA Time    : %.3f ms\n", cuda_ms.count());
    printf("Only kernel execution Time  : %.3f ms\n", launch_ms.count());
    printf("Sequential Time: %.3f ms\n", seq_ms.count());
    printf("Speedup       : %.2fx\n", seq_ms.count() / cuda_ms.count());
    printf("Only kernel Speedup  : %.2fx\n", seq_ms.count() / launch_ms.count());
    printf("Results match : %s\n", correct ? "Yes" : "No");
    printf("------------------------------------------\n");


    // Print first few elements as sample
    printf("\nFirst 10 elements as sample:\n");
    for (int i = 0; i < 10; i++) {
        printf("Index %d: CUDA=%d, Sequential=%d\n", i, C_cuda[i], C_seq[i]);
    }

    // Cleanup
    delete[] A;
    delete[] B;
    delete[] C_cuda;
    delete[] C_seq;


    return 0;
}

