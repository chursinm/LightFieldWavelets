#pragma once

#include <vector>
#include "CudaUtility.h"
#include <sstream>

// TODO refine this
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
	CudaHaarLifting(unsigned int n);
	~CudaHaarLifting();
	void generateData();
	void uploadData();
	GLuint uploadDataGL();
	void downloadData();
	void calculateReference();
	void calculateCuda(); // split, predict, update
	bool checkResult();
	unsigned int size();
	void mapGl();
	void unmapGl();
private:
	std::vector<T> input, cpuOutput, gpuOutput;
	unsigned int m_size;
	T *deviceInput, *deviceWorkBuffer;
	cudaStream_t calcStream;
	cudaGraphicsResource *openglWorkBuffer;
	bool useGL;
};




