#include "stdafx.h"
#include "CudaUtility.h"


float CudaUtility::measurePerformance(std::function<void(void)> functor, unsigned int iterations)
{
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	for(auto i = 0u; i < iterations; ++i)
	{
		functor();
	}
	cudaEventRecord(stop);

	cudaEventSynchronize(stop);
	auto milliseconds = 0.f;
	cudaEventElapsedTime(&milliseconds, start, stop);

	return milliseconds;
}

CudaUtility::CudaUtility()
{
}


CudaUtility::~CudaUtility()
{
}
