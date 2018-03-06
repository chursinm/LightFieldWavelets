#include "stdafx.h"
#include <vector>
#include "CudaHaarLifting.h"
#include "CudaUtility.h"
#include "GLUtility.h"


template <typename T, typename Vector2T>
extern void callSplitPredictUpdate(dim3 grid, dim3 threads, cudaStream_t stream, Vector2T * input, T * oddOut, T * evenOut, T threadcount);

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::generateData(std::function<T (unsigned int index)> generator)
{
	for(auto i = 0u; i < mSize; ++i)
	{
		input[i] = generator(i);
	}
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::uploadData()
{
	const auto halfsize = mSize >> 1;
	auto error = cudaMalloc((void**)&deviceInput, mSize * sizeof(T));
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);

	error = cudaMalloc((void**)&deviceWorkBuffer, (halfsize + mSize) * sizeof(T));
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);

	error = cudaMemcpyAsync(deviceInput, &input[0], mSize * sizeof(T), cudaMemcpyHostToDevice, calcStream);
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
}

template <typename T, typename Vector2T>
GLuint CudaHaarLifting<T, Vector2T>::uploadDataGl()
{
	const auto halfsize = mSize >> 1;
	auto glbuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, (halfsize + mSize) * sizeof(T), GL_DYNAMIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, glbuffer);
	cudaGraphicsGLRegisterBuffer(&openglWorkBuffer, glbuffer, cudaGraphicsRegisterFlagsNone);
	return glbuffer;
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::downloadData()
{
	const auto halfsize = mSize >> 1;
	gpuOutput = std::vector<T>(mSize);

	auto error = cudaMemcpyAsync(&gpuOutput[0], deviceWorkBuffer + halfsize, mSize * sizeof(T), cudaMemcpyDeviceToHost, calcStream);
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
	cudaStreamSynchronize(calcStream);
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::calculateReference()
{
	auto stepInput = input;
	auto stepOutput = std::vector<T>(mSize);
	auto stepsize = mSize;

	while(stepsize > 1)
	{
		const auto halfsize = stepsize >> 1;
		for(auto i = 0u; i < stepsize; i += 2)
		{
			auto odd = stepInput[i + 1] - stepInput[i];
			auto even = stepInput[i] + (odd/2);// (odd >> 1);

			stepOutput[i >> 1] = even;
			stepOutput[halfsize + (i >> 1)] = odd;
		}
		stepsize = halfsize;
		stepInput = stepOutput;
	}
	cpuOutput = stepOutput;
}

template <typename T, typename Vector2T>
std::future<void> CudaHaarLifting<T, Vector2T>::calculateCuda()
{
	return std::move(std::async(std::launch::async, [this]()
	{
		callKernels();
		cudaStreamSynchronize(calcStream);
	}));
}

template<typename T, typename Vector2T>
float CudaHaarLifting<T, Vector2T>::measurePerformanceInMsPerIteration(unsigned int iterations)
{
	auto ms = CudaUtility::measurePerformance([this]()
	{
		callKernels();
	}, iterations);
	return ms / static_cast<float>(iterations);
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::callKernels()
{
	auto threadcount = mSize >> 1;
	auto in = reinterpret_cast<Vector2T*>(deviceInput);
	auto evenOut = deviceWorkBuffer;
	while(threadcount >= 1)
	{
		auto oddOut = deviceWorkBuffer + (mSize >> 1) + threadcount;
		if(threadcount == 1) evenOut = oddOut - 1; // save last even with odds
		// threadcount = blocksInGrid * threadsPerBlock
		// 1 Warp == 32 Threads per Block, es laufen immer 2 Warps parallel. Dazu sollte es Ping/Pong Warps zum verstecken von Latenzen geben
		auto threadsPerBlock = 128u;
		auto blocksInGrid = (threadcount + threadsPerBlock - 1u) / threadsPerBlock; // ceil
		callSplitPredictUpdate<T,Vector2T>(blocksInGrid, threadsPerBlock, calcStream, in, oddOut, evenOut, threadcount);
		threadcount = threadcount >> 1;
		in = reinterpret_cast<Vector2T*>(deviceWorkBuffer);

		//cudaDeviceSynchronize(); // wait to check for errors
		//auto error = cudaGetLastError();
		//if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
	}
}

template <typename T, typename Vector2T>
bool CudaHaarLifting<T, Vector2T>::checkResult()
{
	return memcmp(&cpuOutput[0], &gpuOutput[0], sizeof(T) * mSize) == 0;
}

template <typename T, typename Vector2T>
unsigned int CudaHaarLifting<T, Vector2T>::size() const
{
	return mSize;
}

template <typename T, typename Vector2T>
typename CudaHaarLifting<T, Vector2T>::GlBufferSpec CudaHaarLifting<T, Vector2T>::requiredGlBufferSpec()
{
	const auto halfsize = mSize >> 1;
	return { static_cast<unsigned long long>((halfsize + mSize) * sizeof(T)), reinterpret_cast<void*>(halfsize * sizeof(T)) };
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::mapGl()
{
	auto error = cudaGraphicsMapResources(1, &openglWorkBuffer, calcStream);
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
	size_t size;
	error = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&deviceWorkBuffer), &size, openglWorkBuffer);
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
}

template <typename T, typename Vector2T>
void CudaHaarLifting<T, Vector2T>::unmapGl()
{
	auto error = cudaGraphicsUnmapResources(1, &openglWorkBuffer, calcStream); // really dörty
	if(error != CUDA_SUCCESS) throw CudaHaarLiftingException(cudaGetErrorString(error), __FILE__, __LINE__);
}


template <typename T, typename Vector2T>
CudaHaarLifting<T, Vector2T>::CudaHaarLifting(unsigned int poweroftwo): mSize(std::pow(2, poweroftwo)),
                                                                        openglWorkBuffer(nullptr),
                                                                        input(std::pow(2, poweroftwo))
{
	cudaStreamCreateWithFlags(&calcStream, cudaStreamNonBlocking);
	//calcStream = 0;
}


template <typename T, typename Vector2T>
CudaHaarLifting<T, Vector2T>::~CudaHaarLifting()
{
	cudaFree(deviceInput);
	cudaFree(deviceWorkBuffer);
}



