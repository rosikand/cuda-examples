#include <iostream>
#include <cstdio>




// ------------------ Helpers ------------------
void initVector(int* vecPtr, int vecLen) {
    if (vecPtr) {
        for (int i=0; i < vecLen; i++) {
            vecPtr[i] += i + 1;
        }
    }
}


void printVector(int* vecPtr, int vecLen) {
    if (vecPtr) {
        for (int i=0; i < vecLen; i++) {
            printf("vec[%d] = %d\n", i, vecPtr[i]);
        }
    }
}


int compress_idx(int i, int j, int numCols) {
    // takes 2D idx [i][j] and calculates its compressed, 1D array index
    return i * numCols + j;
}


// non-CUDA version
int dotProd(int* vecOne, int* vecTwo, int vecLen) {
    int accumulator = 0;
    for (int i = 0; i < vecLen; i++) {
        accumulator += vecOne[i] * vecTwo[i];
    }

    return accumulator;
}

// ------------------ CUDA kernels ------------------


__global__ void helloWorldKernel() {
    printf("Hello, World from thread %d in block %d!\n", threadIdx.x, blockIdx.x);
}

__global__ void dotProdSequentialKernel(int *vecOne, int *vecTwo, int *result, int vecLen) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Thread idx: %d \n", idx);

    int accumulator = 0;
    for (int i = 0; i < vecLen; i++) {
        accumulator += vecOne[i] * vecTwo[i];
    }

    printf("From device, the result is: %d \n", accumulator);
    *result = accumulator;
    __syncthreads();
}

// ---------------------------------------------

int main() {

    // we will start with 1D vector/array dot products since that is easier
    // make arrays (vectors)
    int vecLen = 3;
    int *vecOne = (int*)malloc(vecLen * sizeof(int));
    int *vecTwo = (int*)malloc(vecLen * sizeof(int));

    initVector(vecOne, vecLen);
    initVector(vecTwo, vecLen);
    initVector(vecTwo, vecLen);
    printVector(vecOne, vecLen);
    printVector(vecTwo, vecLen);

    int res = dotProd(vecOne, vecTwo, vecLen);
    std::cout << res << std::endl;

    // now lets do it with CUDA
    int *vecOneDevice;
    int *vecTwoDevice;
    int *resultDevice;
    int resultHost = 0;
    cudaMalloc(&vecOneDevice, vecLen * sizeof(int));
    cudaMalloc(&vecTwoDevice, vecLen * sizeof(int));
    cudaMalloc(&resultDevice, sizeof(int));
    cudaMemcpy(vecOneDevice, vecOne, vecLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(vecTwoDevice, vecTwo, vecLen * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(resultDevice, &resultHost, sizeof(int), cudaMemcpyHostToDevice);

    // launch kernel
    dotProdSequentialKernel<<<1,1>>>(vecOneDevice, vecTwoDevice, resultDevice, vecLen);

    // copy back to the host
    cudaMemcpy(&resultHost, resultDevice, sizeof(int), cudaMemcpyDeviceToHost);
    printf("From host, the result is: %d \n", resultHost);

    // Wait for the GPU to finish
    cudaDeviceSynchronize();

    // Clean up memory
    free(vecOne);
    free(vecTwo);

    std::cout << "Hello world from CPU!" << std::endl;
    return 0;
}
