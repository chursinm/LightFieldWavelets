#pragma once

#include <vector>

__global__ void split(void);
__global__ void predict(void);
__global__ void update(void);

class CudaHaarLifting
{
public:
	CudaHaarLifting(unsigned int n);
	~CudaHaarLifting();
	void generateData();
	void uploadData();
	void downloadData();
	void calculateReference();
	void calculateCuda(); // split, predict, update
private:
	std::vector<int> input, cpuOutput, gpuOutput, gpuEven, gpuOdd;
	unsigned int size;
	int *deviceMemory, *deviceEven, *deviceOdd;
};




