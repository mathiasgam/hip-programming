#include <hip/hip_runtime.h>

#include <cstdlib>
#include <vector>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

#define LOG2SIZE 12
const static int width = 1<<LOG2SIZE;
const static int height = 1<<LOG2SIZE;

const static int tile_dim_x = 16;
const static int tile_dim_y = 16;

__global__ void copy_kernel(float *in, float *out, size_t width, size_t height, int stride) {
  size_t x_index = blockIdx.x * tile_dim_x + threadIdx.x;
  size_t y_index = blockIdx.y * tile_dim_y + threadIdx.y;

  size_t index = (y_index * width + stride*x_index) % (width*height);
  size_t index_in = (y_index * width + stride*x_index) % (width*height);

  out[index] = in[index_in];
}

int main() {
  std::vector<float> matrix_in;
  std::vector<float> matrix_out;

  matrix_in.resize(width * height);
  matrix_out.resize(width * height);

  for (int i = 0; i < width * height; i++) {
    matrix_in[i] = (float)rand() / (float)RAND_MAX;
  }

  float *d_in;
  float *d_out;

  HIP_ERRCHK(hipMalloc((void **)&d_in, (width * height) * sizeof(float)));
  HIP_ERRCHK(hipMalloc((void **)&d_out, (width * height) * sizeof(float)));

  HIP_ERRCHK(hipMemcpy(d_in, matrix_in.data(), width * height * sizeof(float),
            hipMemcpyHostToDevice));

  printf("Setup complete. Launching kernel \n");
  int block_x = width / tile_dim_x;
  int block_y = height / tile_dim_y;
  

  // Create events

  /* printf("Warm up the gpu!\n"); */
  /* for(int i=1;i<=10;i++){ */
  /*   hipLaunchKernelGGL(copy_kernel, dim3(block_x, block_y), */
  /*                     dim3(tile_dim_x, tile_dim_y), 0, 0, d_in, d_out, width, */
  /*                     height);} */


  
  for(int i=1;i<=21;i++){
    hipLaunchKernelGGL(copy_kernel, dim3(block_x, block_y), dim3(tile_dim_x, tile_dim_y), 0, 0, d_in, d_out, width, height, (1<<i)-1);
  }
  

  HIP_ERRCHK(hipDeviceSynchronize());
  float time_kernel;

  printf("Done!\n");
  HIP_ERRCHK(hipMemcpy(matrix_out.data(), d_out, width * height * sizeof(float),
            hipMemcpyDeviceToHost));


  return 0;
}
