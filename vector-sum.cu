#include <stdio.h>
#include <cuda_runtime.h>
#include <iostream>

// Kernel function to print "Hello, World!" from each thread
__global__ void vectorSumKernel(const int* A, const int* B, const int* C, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        C[idx] = A[idx] + B[idx];
    }

}

int main() {
    // define arrays 
    int n = 10; 
    int A[] = {1,2,3,4,5,6,7,8,9,10};
    int B[] = {10,20,30,40,50,60,70,80,90,100};
    int C[n];


    // send over to gpu 
    int* a_ptr;
    cudaMalloc(&a_ptr, n * sizeof(int));
    cudaMemcpy(a_ptr, A, n * sizeof(int), cudaMemcpyHostToDevice); 

    int* b_ptr;
    cudaMalloc(&b_ptr, n * sizeof(int));
    cudaMemcpy(b_ptr, B, n * sizeof(int), cudaMemcpyHostToDevice);

    int* c_ptr;
    cudaMalloc(&c_ptr, n * sizeof(int));

    // Launch the kernel with 1 block of 10 threads
    vectorSumKernel<<<1, 10>>>(a_ptr, b_ptr, c_ptr, n);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(C, c_ptr, n * sizeof(int), cudaMemcpyDeviceToHost);

    // print array
    for (int i = 0; i < n; i++) {
        std::cout << "C[" << i << "] = " << C[i] << std::endl;
    }


    cudaFree(a_ptr);
    cudaFree(b_ptr);
    cudaFree(c_ptr);

    return 0;
}
