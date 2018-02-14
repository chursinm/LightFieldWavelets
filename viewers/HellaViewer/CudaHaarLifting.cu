#include "stdafx.h"
#include <vector>
#include "CudaHaarLifting.h"
#include "CudaUtility.h"

__global__ void splitPredictUpdate(int2 * input, int * oddOut, int * evenOut)
{
	auto gid = blockIdx.x * blockDim.x + threadIdx.x;

	// split
	auto evenOdd = input[gid];
	auto evenI = evenOdd.x;
	auto oddI = evenOdd.y;

	// predict
	oddI -= evenI;

	// update
	evenI += (oddI >> 1);

	// store result
	oddOut[gid] = oddI;
	evenOut[gid] = evenI;
}

void CudaHaarLifting::generateData()
{
	auto whatever = [](unsigned int index) { return index * 2 + 3; };
	for(auto i = 0u; i < size; ++i)
	{
		input[i] = whatever(i);
	}
}

void CudaHaarLifting::uploadData()
{
	auto error = cudaMalloc((void**)&deviceInput, size * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMalloc((void**)&deviceOutputEven, size * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMalloc((void**)&deviceOutputOdd, size * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMemcpy(deviceInput, &input[0], size * sizeof(int), cudaMemcpyHostToDevice);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);
}

void CudaHaarLifting::downloadData()
{
	gpuOutputEven = std::vector<int>(size);
	gpuOutputOdd = std::vector<int>(size);
	gpuOutput = std::vector<int>(size);

	auto error = cudaMemcpy(&gpuOutputEven[0], deviceOutputEven, size * sizeof(int), cudaMemcpyDeviceToHost);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMemcpy(&gpuOutputOdd[0], deviceOutputOdd, size * sizeof(int), cudaMemcpyDeviceToHost);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	gpuOutput = gpuOutputOdd;
	gpuOutput[0] = gpuOutputEven[0];
}

void CudaHaarLifting::calculateReference()
{
	auto stepInput = input;
	auto stepOutput = std::vector<int>(size);
	auto stepsize = size;

	while(stepsize > 1)
	{
		auto halfsize = stepsize >> 1;
		for(auto i = 0u; i < stepsize; i += 2)
		{
			auto odd = stepInput[i + 1] - stepInput[i];
			auto even = stepInput[i] + (odd >> 1);

			stepOutput[i >> 1] = even;
			stepOutput[halfsize + (i >> 1)] = odd;
		}
		stepsize = halfsize;
		stepInput = stepOutput;
	}
	cpuOutput = stepOutput;
}

void CudaHaarLifting::calculateCuda()
{
	auto threadcount = size >> 1;
	auto in = (int2*)deviceInput;
	auto oddOut = deviceOutputOdd + threadcount;
	auto evenOut = deviceOutputEven;

	auto ms = CudaUtility::measurePerformance([&]()
	{
		while(threadcount >= 1)
		{
			splitPredictUpdate KERNEL_ARGS2(threadcount, 1) (in, oddOut, evenOut);
			threadcount = threadcount >> 1;
			in = (int2*)deviceOutputEven;
			oddOut = deviceOutputOdd + threadcount;
			evenOut = deviceOutputEven;
		}
	}, 1000);

	// 6.8 is reference time; 5.35 with reinterpret cast
	std::cout << "Executing b for 1000 iterations took " << ms/1000.f << " ms on average" << std::endl;
}

bool CudaHaarLifting::checkResult()
{
	return memcmp(&cpuOutput[0], &gpuOutput[0], sizeof(int) * size) == 0;
}


CudaHaarLifting::CudaHaarLifting(unsigned int poweroftwo): size(std::pow(2, poweroftwo)), input(std::pow(2,poweroftwo))
{
}


CudaHaarLifting::~CudaHaarLifting()
{
	cudaFree(deviceInput);
}


