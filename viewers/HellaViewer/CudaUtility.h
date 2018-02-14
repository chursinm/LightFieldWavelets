#pragma once

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#ifndef CUDA_SUCCESS
#define CUDA_SUCCESS 0
#endif

class CudaUtility
{
public:
	/// Executes functor iterations times and returns the exectution time on the gpu in milliseconds.
	/// see: cudaEventElapsedTime
	static float measurePerformance(std::function<void (void)> functor, unsigned int iterations);
private:
	CudaUtility();
	~CudaUtility();
};

