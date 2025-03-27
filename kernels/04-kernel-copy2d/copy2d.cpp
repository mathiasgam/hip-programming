#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <vector>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

template<typename T>
inline T div_up(T a, T b) {
    return (a + b - 1) / b;
}

// Copy all elements using threads in a 2D grid
__global__ void copy2d(double* dst, double* src, size_t nx, size_t ny) {
    size_t ix = threadIdx.x + blockIdx.x * blockDim.x;
    size_t iy = threadIdx.y + blockIdx.y * blockDim.y;

    if (ix < nx && iy < ny) {
        // We're computing 1D index from a 2D index and copying from src to dst
        const size_t index = ix + iy * nx;
        dst[index] = src[index];
    }
}

int main() {
    static constexpr size_t num_cols = 600;
    static constexpr size_t num_rows = 400;
    static constexpr size_t num_values = num_cols * num_rows;
    static constexpr size_t num_bytes = sizeof(double) * num_values;
    std::vector<double> x(num_values);
    std::vector<double> y(num_values, 0.0);

    // Initialise data
    for (size_t i = 0; i < num_values; i++) {
        x[i] = static_cast<double>(i) / 1000.0;
    }

    // TODO: Allocate + copy initial values to GPU
    double *dx, *dy;
    HIP_ERRCHK(hipMalloc(&dx, num_bytes));
    HIP_ERRCHK(hipMalloc(&dy, num_bytes));
    HIP_ERRCHK(hipMemcpy(dx, x.data(), num_bytes, hipMemcpyHostToDevice));

    // TODO: Define grid dimensions
    // Use dim3 structure for threads and blocks
    size_t block_width = 32;
    size_t block_height = 32;
    dim3 block_size(32,32,1);
    printf("block_size: [%d,%d,%d]\n", block_size.x, block_size.y, block_size.z);

    size_t grid_width = div_up(num_cols, block_width);
    size_t grid_height = div_up(num_rows, block_height);
    dim3 grid_size(grid_width, grid_height, 1);
    printf("grid_size: [%d,%d,%d]\n", grid_size.x, grid_size.y, grid_size.z);

    printf("global_size: [%d,%d,%d]\n", grid_size.x * block_size.x, grid_size.y * block_size.y, grid_size.z * block_size.z);


    // TODO: launch the device kernel
    copy2d<<<grid_size, block_size>>>(dy, dx, num_cols, num_rows);

    // TODO: Copy results back to the CPU vector y
    HIP_ERRCHK(hipMemcpy(y.data(), dy, num_bytes, hipMemcpyDeviceToHost));

    HIP_ERRCHK(hipDeviceSynchronize());

    // TODO: Free device memory
    HIP_ERRCHK(hipFree(dx));
    HIP_ERRCHK(hipFree(dy));

    // Check result of computation on the GPU
    double error = 0.0;
    for (size_t i = 0; i < num_values; i++) {
        error += abs(x[i] - y[i]);
    }

    printf("total error: %f\n", error);
    printf("  reference: %f at (42,42)\n", x[42 * num_rows + 42]);
    printf("     result: %f at (42,42)\n", y[42 * num_rows + 42]);

    return 0;
}
