#include <cstdio>
#include <time.h>
#include <hip/hip_runtime.h>
#include <chrono>

#define get_mus(X) std::chrono::duration_cast<std::chrono::microseconds>(X).count()
#define chrono_clock std::chrono::high_resolution_clock::now()

/* A simple GPU kernel definition */
__global__ void kernel(int *d_a, int n_total)
{
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx < n_total)
    d_a[idx] = idx;
}

/* The main function */
int main(){
  
  // Problem size
  constexpr int n_total = 1<<22;

  // Device grid sizes
  constexpr int blocksize = 256;
  constexpr int gridsize = (n_total - 1 + blocksize) / blocksize;

  // Allocate host and device memory
  int *a, *d_a;
  const int bytes = n_total * sizeof(int);
  hipHostMalloc((void**)&a, bytes); // host pinned
  hipMalloc((void**)&d_a, bytes);   // device pinned

  // Create events
  hipEvent_t event_start, event_copy, event_end;
  hipEventCreate(&event_start);
  hipEventCreate(&event_copy);
  hipEventCreate(&event_end);

  // Create stream
  hipStream_t stream;
  hipStreamCreate(&stream);

  // Start timed GPU kernel and device-to-host copy
  auto start_kernel_clock = chrono_clock;
  hipEventRecord(event_start, stream);
  kernel<<<gridsize, blocksize, 0, stream>>>(d_a, n_total);

  auto start_d2h_clock = chrono_clock;
  hipEventRecord(event_copy, stream);
  hipMemcpyAsync(a, d_a, bytes, hipMemcpyDeviceToHost, stream);

  auto stop_clock = chrono_clock;
  hipEventRecord(event_end, stream);
  hipStreamSynchronize(stream);

  // Exctract elapsed timings from event recordings
  float time_kernel, time_copy, time_total;
  hipEventElapsedTime(&time_kernel, event_start, event_copy);
  hipEventElapsedTime(&time_copy, event_copy, event_end);
  hipEventElapsedTime(&time_total, event_start, event_end);

  // Check that the results are right
  int error = 0;
  for(int i = 0; i < n_total; ++i){
    if(a[i] != i)
      error = 1;
  }

  // Print results
  if(error)
    printf("Results are incorrect!\n");
  else
    printf("Results are correct!\n");

  // Print event timings
  printf("Event timings:\n");
  // #error print event timings here
  printf("  %.3f ms - kernel\n", time_kernel);
  printf("  %.3f ms - device to host copy\n", time_copy);
  printf("  %.3f ms - total time\n", time_total);


  // Print clock timings
  printf("clock_t timings:\n");
  printf("  %.3f ms - kernel\n", 1e3 * (double)get_mus(start_d2h_clock - start_kernel_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - device to host copy\n", 1e3 * (double)get_mus(stop_clock - start_d2h_clock) / CLOCKS_PER_SEC);
  printf("  %.3f ms - total time\n", 1e3 * (double)get_mus(stop_clock - start_kernel_clock) / CLOCKS_PER_SEC);

  // Destroy Stream
  hipStreamDestroy(stream);

  // Destroy events
  hipEventDestroy(event_start);
  hipEventDestroy(event_copy);
  hipEventDestroy(event_end);

  // Deallocations
  hipFree(d_a); // Device
  hipHostFree(a); // Host
}
