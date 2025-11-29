# CUDA at Scale: High-Throughput Sobel Edge Detection

## Project Overview
This project implements a GPU-accelerated image processing pipeline using CUDA and C++. [cite_start]It is designed to perform batch processing on a large set of images ("large amount of data") by applying a Sobel Edge Detection filter[cite: 3].

[cite_start]The application simulates an enterprise-scale edge computing scenario where raw image data (PGM format) is ingested, processed in parallel on the GPU to extract features (edges), and saved to storage[cite: 10].

## Repository Structure
[cite_start]This repository meets the requirements for a valid code repository[cite: 2, 6, 8]:

* **`src/`**: Contains the source code.
  * `main.cpp`: Host code handling file I/O and CLI arguments.
  * `sobel.cu`: CUDA device code containing the `SobelKernel`.
  * `sobel.h`: Header file for kernel declarations.
* **`data/`**: Input directory containing raw PGM images.
* **`output/`**: Destination directory for the processed edge-detected images.
* [cite_start]**`Makefile`**: Script to compile the project using `nvcc` and manage build artifacts[cite: 8].

## Prerequisites
* **Hardware**: NVIDIA GPU (Compute Capability 3.0 or higher).
* **Software**:
  * CUDA Toolkit (nvcc).
  * GCC/G++ Compiler.
  * Make.

## Compilation
To build the project, navigate to the root directory and run:

```bash
make

This will generate an executable named sobel_filter. To remove compiled objects and the executable, run:

Bash

make clean
Usage
The application uses a Command Line Interface (CLI) that accepts two arguments: the input directory and the output directory.

Syntax:

Bash

./sobel_filter <input_directory> <output_directory>
Example Execution:

Bash

# 1. Ensure the output directory exists
mkdir -p output

# 2. Run the processor
./sobel_filter ./data ./output
Expected Output: The terminal will display the status of each file being processed:

Plaintext

Processing: texture1.pgm
Processing: texture2.pgm
...
