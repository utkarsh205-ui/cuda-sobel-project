#ifndef SOBEL_H_
#define SOBEL_H_

#include <cuda_runtime.h>
#include <string>

// Applies Sobel Edge Detection on the GPU
// width: Image width
// height: Image height
// d_input: Device pointer to input data
// d_output: Device pointer to output data
void DispatchSobel(int width, int height, unsigned char* d_input, unsigned char* d_output);

#endif  // SOBEL_H_