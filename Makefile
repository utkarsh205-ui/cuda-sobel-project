NVCC = nvcc
CXX = nvcc
NVCC_FLAGS = -I./src -O2
CXX_FLAGS = -I./src -O2

SRCS_CU = src/sobel.cu
SRCS_CPP = src/main.cpp
OBJS = src/sobel.o src/main.o
TARGET = sobel_filter

all: $(TARGET)

$(TARGET): $(OBJS)
	$(NVCC) $(OBJS) -o $(TARGET)

src/sobel.o: src/sobel.cu
	$(NVCC) $(NVCC_FLAGS) -c $< -o $@

src/main.o: src/main.cpp
	$(CXX) $(CXX_FLAGS) -c $< -o $@

clean:
	rm -f src/*.o $(TARGET)
