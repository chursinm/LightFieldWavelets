#pragma once
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <vector>
class CudaPropertyViewer
{
public:
	CudaPropertyViewer();
	~CudaPropertyViewer();
	void print();
	std::vector<cudaDeviceProp> props;
};

