#pragma once

#include <vector>


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
	bool checkResult();
private:
	std::vector<int> input, cpuOutput, gpuOutput, gpuOutputEven, gpuOutputOdd;
	unsigned int size;
	int *deviceInput, *deviceOutputEven, *deviceOutputOdd;
};




