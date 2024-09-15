#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 1024  // Matrix size (N x N)
#define BLOCK_SIZE 32

// CUDA kernel for matrix dot product
__global__ void matrixDotProduct(float *A, float *B, float *C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;

    if (row < n && col < n) {
        for (int i = 0; i < n; i++) {
            sum += A[row * n + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

// Function to initialize a matrix with random float values
void initMatrix(float *matrix, int n) {
    for (int i = 0; i < n * n; i++) {
        matrix[i] = (float)rand() / RAND_MAX;
    }
}

int main() {
    float *h_A, *h_B, *h_C;  // Host matrices
    float *d_A, *d_B, *d_C;  // Device matrices
    size_t bytes = N * N * sizeof(float);

    // Allocate host memory
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C = (float*)malloc(bytes);

    // Initialize host matrices
    initMatrix(h_A, N);
    initMatrix(h_B, N);

    // Allocate device memory
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // Copy host memory to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocksPerGrid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch CUDA kernel
    matrixDotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Copy result from device to host
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // Verify result (check a few elements)
    printf("Verification (first few elements):\n");
    for (int i = 0; i < 5; i++) {
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += h_A[i * N + j] * h_B[j * N + i];
        }
        printf("C[%d][%d]: GPU = %f, CPU = %f\n", i, i, h_C[i * N + i], sum);
    }

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
