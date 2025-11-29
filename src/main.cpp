#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <dirent.h>
#include <cuda_runtime.h>
#include "sobel.h"

// Helper to read PGM images (Grayscale)
unsigned char* ReadPGM(const std::string& filename, int& width, int& height) {
    std::ifstream file(filename, std::ios::binary);
    std::string line;
    std::getline(file, line); // P5
    while(file.peek() == '#') std::getline(file, line); // Skip comments
    file >> width >> height;
    int maxVal;
    file >> maxVal;
    file.ignore(256, '\n'); // Skip single whitespace

    unsigned char* data = new unsigned char[width * height];
    file.read(reinterpret_cast<char*>(data), width * height);
    return data;
}

// Helper to write PGM images
void WritePGM(const std::string& filename, unsigned char* data, int width, int height) {
    std::ofstream file(filename, std::ios::binary);
    file << "P5\n" << width << " " << height << "\n255\n";
    file.write(reinterpret_cast<char*>(data), width * height);
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_dir> <output_dir>" << std::endl;
        return 1;
    }

    std::string inputDir = argv[1];
    std::string outputDir = argv[2];

    DIR* dir;
    struct dirent* ent;
    if ((dir = opendir(inputDir.c_str())) != NULL) {
        while ((ent = readdir(dir)) != NULL) {
            std::string fileName = ent->d_name;
            if (fileName.find(".pgm") != std::string::npos) {
                std::cout << "Processing: " << fileName << std::endl;

                int width, height;
                std::string fullPath = inputDir + "/" + fileName;
                unsigned char* h_input = ReadPGM(fullPath, width, height);
                unsigned char* h_output = new unsigned char[width * height];

                // Allocate Device Memory
                unsigned char *d_input, *d_output;
                cudaMalloc(&d_input, width * height);
                cudaMalloc(&d_output, width * height);

                // Transfer to Device
                cudaMemcpy(d_input, h_input, width * height, cudaMemcpyHostToDevice);

                // Run Kernel
                DispatchSobel(width, height, d_input, d_output);
                cudaDeviceSynchronize();

                // Transfer back
                cudaMemcpy(h_output, d_output, width * height, cudaMemcpyDeviceToHost);

                // Save
                std::string outPath = outputDir + "/edge_" + fileName;
                WritePGM(outPath, h_output, width, height);

                // Cleanup
                delete[] h_input;
                delete[] h_output;
                cudaFree(d_input);
                cudaFree(d_output);
            }
        }
        closedir(dir);
    } else {
        std::cerr << "Could not open directory" << std::endl;
        return 1;
    }

    return 0;
}