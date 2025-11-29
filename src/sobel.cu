#include "sobel.h"
#include <math.h>
#include <iostream>

// Kernel to calculate Sobel Edge Detection
__global__ void SobelKernel(unsigned char* input, unsigned char* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x > 0 && x < width - 1 && y > 0 && y < height - 1) {
        // Sobel kernels
        int Gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
        int Gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

        float sumX = 0.0f;
        float sumY = 0.0f;

        // Convolve
        for (int i = -1; i <= 1; i++) {
            for (int j = -1; j <= 1; j++) {
                int pixelVal = input[(y + i) * width + (x + j)];
                sumX += pixelVal * Gx[i + 1][j + 1];
                sumY += pixelVal * Gy[i + 1][j + 1];
            }
        }

        // Calculate magnitude
        int magnitude = (int)sqrtf(sumX * sumX + sumY * sumY);
        if (magnitude > 255) magnitude = 255;
        if (magnitude < 0) magnitude = 0;

        output[y * width + x] = (unsigned char)magnitude;
    }
}

void DispatchSobel(int width, int height, unsigned char* d_input, unsigned char* d_output) {
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);

    SobelKernel<<<gridSize, blockSize>>>(d_input, d_output, width, height);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(err) << std::endl;
    }
}