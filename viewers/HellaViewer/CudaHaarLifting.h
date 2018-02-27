#pragma once

#include <vector>
#include "CudaUtility.h"
#include <sstream>

// As this is our playground to test the cuda api, every method is public.
// TODO encapsulate, once we do not need the playground anymore
class CudaHaarLiftingException : public std::exception
{
public:
	explicit CudaHaarLiftingException(const std::string& fault, const std::string& fileName, const unsigned int lineNumber) : mFault(fault), mLineNumber(lineNumber), mFileName(fileName)
	{
		std::stringstream wtf;
		wtf << "CudaHaarLiftingException at " << mFileName << ":" << mLineNumber << ": " << mFault;
		mMessage = wtf.str();
	}
	char const * what() const override
	{
		return mMessage.c_str();
	}
private:
	std::string mFault;
	unsigned int mLineNumber;
	std::string mFileName;
	std::string mMessage;
};
template <typename T, typename Vector2T>
class CudaHaarLifting
{
public:
	struct GlBufferSpec
	{
		unsigned long long mByteSize;
		void* mByteOffset;
	};

	CudaHaarLifting(unsigned int n);
	~CudaHaarLifting();
	void generateData(std::function<T (unsigned int index)> generator = [](auto index) { return static_cast<T>(index); });
	void uploadData();
	GLuint uploadDataGl();
	void downloadData();
	void calculateReference();
	std::future<void> calculateCuda(); // split, predict, update
	float measurePerformanceInMsPerIteration(unsigned int iterations);
	bool checkResult();
	unsigned int size() const;
	
	
	GlBufferSpec requiredGlBufferSpec();
	void mapGl();
	void unmapGl();
private:
	void callKernels();
	std::vector<T> input, cpuOutput, gpuOutput;
	unsigned int mSize;
	T *deviceInput, *deviceWorkBuffer;
	cudaStream_t calcStream;
	cudaGraphicsResource *openglWorkBuffer;
};




