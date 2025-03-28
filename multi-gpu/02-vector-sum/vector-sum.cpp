#include <cstdio>
#include <cmath>
#include <hip/hip_runtime.h>

#define HIP_ERRCHK(result) (hip_errchk(result, __FILE__, __LINE__))
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file,
               line);
        exit(EXIT_FAILURE);
    }
}

// Data structure for storing decomposition information
struct Decomp {
    int len;    // length of the array for the current device
    int start;  // start index for the array on the current device
};


/* HIP kernel for the addition of two vectors, i.e. C = A + B */
__global__ void vector_add(double *C, const double *A, const double *B, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Do not try to access past the allocated memory
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}


int main(int argc, char *argv[])
{
    const int ThreadsInBlock = 128;
    double *dA[2], *dB[2], *dC[2];
    double *hA, *hB, *hC;
    int devicecount;
    int N = 1000000;
    hipEvent_t start, stop;
    hipStream_t strm[2];
    Decomp dec[2];

    // TODO: Check that we have two HIP devices available
    HIP_ERRCHK(hipGetDeviceCount(&devicecount));
    if (devicecount < 2) {
        fprintf(stderr, "Not enough devices allocated!, expected > 2, got %d\n", devicecount);
        return EXIT_FAILURE;
    }


    // Create timing events
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventCreate(&start));
    HIP_ERRCHK(hipEventCreate(&stop));

    // Allocate host memory
    // TODO: Allocate enough pinned host memory for hA, hB, and hC
    //       to store N doubles each
    {
        size_t alloc_size = sizeof(double) * N;
        HIP_ERRCHK(hipHostMalloc(&hA, alloc_size));
        HIP_ERRCHK(hipHostMalloc(&hB, alloc_size));
        HIP_ERRCHK(hipHostMalloc(&hC, alloc_size));
    }

    // Initialize host memory
    for(int i = 0; i < N; ++i) {
        hA[i] = 1.0;
        hB[i] = 2.0;
    }

    // Decomposition of data for each stream
    dec[0].len   = N / 2;
    dec[0].start = 0;
    dec[1].len   = N - N / 2;
    dec[1].start = dec[0].len;

    // Allocate memory for the devices and per device streams
    for (int i = 0; i < 2; ++i) {
        // TODO: Allocate enough device memory for dA[i], dB[i], dC[i]
        //       to store dec[i].len doubles
        // TODO: Create a stream for each device
        HIP_ERRCHK(hipSetDevice(i));

        size_t alloc_size = sizeof(double) * dec[i].len;
        HIP_ERRCHK(hipMalloc(&dA[i], alloc_size));
        HIP_ERRCHK(hipMalloc(&dB[i], alloc_size));
        HIP_ERRCHK(hipMalloc(&dC[i], alloc_size));

        HIP_ERRCHK(hipStreamCreate(&strm[i]));
    }

    // Start timing
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventRecord(start));

    /* Copy each decomposed part of the vectors from host to device memory
       and execute a kernel for each part.
       Note: one needs to use streams and asynchronous calls! Without this
       the execution is serialized because the memory copies block the
       execution of the host process. */
    for (int i = 0; i < 2; ++i) {
        // TODO: Set active device
        HIP_ERRCHK(hipSetDevice(i));
        // TODO: Copy data from host to device asynchronously (hA[dec[i].start] -> dA[i], hB[dec[i].start] -> dB[i])
        HIP_ERRCHK(hipMemcpyAsync(dA[i], &hA[dec[i].start], sizeof(double) * dec[i].len, hipMemcpyHostToDevice, strm[i]));
        HIP_ERRCHK(hipMemcpyAsync(dB[i], &hB[dec[i].start], sizeof(double) * dec[i].len, hipMemcpyHostToDevice, strm[i]));
        // TODO: Launch 'vector_add()' kernel to calculate dC = dA + dB
        int num_blocks = (dec[i].len + ThreadsInBlock - 1) / ThreadsInBlock;
        vector_add<<<num_blocks, ThreadsInBlock, 0, strm[i]>>>(dC[i], dA[i], dB[i], dec[i].len);
        // TODO: Copy data from device to host (dC[i] -> hC[dec[0].start])
        HIP_ERRCHK(hipMemcpyAsync(&hC[dec[i].start], dC[i], sizeof(double) * dec[i].len, hipMemcpyDeviceToHost, strm[i]));
    }

    // Synchronize and destroy the streams
    for (int i = 0; i < 2; ++i) {
        // TODO: Add synchronization calls and destroy streams
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipStreamSynchronize(strm[i]));
    }

    // Stop timing
    // TODO: Add here the timing event stop calls
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventRecord(stop));

    // Free device memory
    for (int i = 0; i < 2; ++i) {
        // TODO: Deallocate device memory
        HIP_ERRCHK(hipSetDevice(i));
        HIP_ERRCHK(hipFree(dA[i]));
        HIP_ERRCHK(hipFree(dB[i]));
        HIP_ERRCHK(hipFree(dC[i]));
    }

    // Check results
    int errorsum = 0;
    for (int i = 0; i < N; i++) {
        errorsum += hC[i] - 3.0;
    }
    printf("Error sum = %i\n", errorsum);

    // Calculate the elapsed time
    float gputime;
    HIP_ERRCHK(hipSetDevice(0));
    HIP_ERRCHK(hipEventElapsedTime(&gputime, start, stop));
    printf("Time elapsed: %f\n", gputime / 1000.);

    // Deallocate host memory
    HIP_ERRCHK(hipHostFree((void*)hA));
    HIP_ERRCHK(hipHostFree((void*)hB));
    HIP_ERRCHK(hipHostFree((void*)hC));

    return 0;
}
