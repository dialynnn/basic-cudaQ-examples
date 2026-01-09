#pragma once
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

// Wrapper declaration (implemented in cuda_kernel.cu)
void launch_basic_kernel(int *d_out, int value);

#ifdef __cplusplus
}
#endif