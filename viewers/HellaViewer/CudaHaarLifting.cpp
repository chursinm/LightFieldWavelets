#include "stdafx.h"
#include <vector>
#include "CudaHaarLifting.h"

__global__ void split(int *input, int *even, int *odd)
{
	int tid = blockIdx.x;
	even[tid] = input[tid * 2];
	odd[tid] = input[tid * 2 + 1];
}
__global__ void predict(int * even, int * odd)
{
	int tid = blockIdx.x;
}
__global__ void update(void)
{

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
	auto error = cudaMalloc((void**)&deviceMemory, size * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMalloc((void**)&deviceEven, size / 2 * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMalloc((void**)&deviceOdd, size / 2 * sizeof(int));
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMemcpy(deviceMemory, &input[0], size * sizeof(int), cudaMemcpyHostToDevice);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);
}

void CudaHaarLifting::downloadData()
{
	auto halfsize = size >> 1;
	gpuOutput = std::vector<int>(size);
	gpuEven = std::vector<int>(halfsize);
	gpuOdd = std::vector<int>(halfsize);

	auto error = cudaMemcpy(&gpuOutput[0], deviceMemory, size * sizeof(int), cudaMemcpyDeviceToHost);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMemcpy(&gpuEven[0], deviceEven, halfsize * sizeof(int), cudaMemcpyDeviceToHost);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);

	error = cudaMemcpy(&gpuOdd[0], deviceOdd, halfsize * sizeof(int), cudaMemcpyDeviceToHost);
	if(error != CUDA_SUCCESS) throw cudaGetErrorString(error);
}

void CudaHaarLifting::calculateReference()
{
}

void CudaHaarLifting::calculateCuda()
{
	unsigned int threadcount = size / 2;
	split << <threadcount, 1 >> >(deviceMemory, deviceEven, deviceOdd);


}

CudaHaarLifting::CudaHaarLifting(unsigned int n): size(n), input(n)
{
	if(size % 2 != 0) throw "size needs to be dividable by 2";
}


CudaHaarLifting::~CudaHaarLifting()
{
	cudaFree(deviceMemory);
}


