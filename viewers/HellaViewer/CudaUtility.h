#pragma once
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

