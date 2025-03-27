#include <hip/hip_runtime.h>
#include <stdio.h>

#define HIP_ERRCHK(result) hip_errchk(result, __FILE__, __LINE__)
static inline void hip_errchk(hipError_t result, const char *file, int line) {
    if (result != hipSuccess) {
        printf("\n\n%s in %s at line %d\n", hipGetErrorString(result), file, line);
        exit(EXIT_FAILURE);
    }
}

int main() {
    // There's a bug in this program, find out what it is by implementing the
    // function above, and correct it
    int count = 0;
    HIP_ERRCHK(hipGetDeviceCount(&count));
    // HIP_ERRCHK(hipSetDevice(count)); // Cannot set device to count, as the valid id's go from 0 to (not including) count.
    HIP_ERRCHK(hipSetDevice(count - 1)); // Set to the last device.

    int device = 0;
    HIP_ERRCHK(hipGetDevice(&device));

    printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);

    return 0;
}
