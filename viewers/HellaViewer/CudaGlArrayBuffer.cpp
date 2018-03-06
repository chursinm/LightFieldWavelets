#include "stdafx.h"
#include "CudaGlArrayBuffer.h"
#include "GLUtility.h"
#include "CudaUtility.h"

template <typename T>
CudaGlArrayBuffer<T>::CudaGlArrayBuffer(unsigned int elementCount, T * data): mCudaMapped(false)
{
	mGlBuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, elementCount, data, GL_DYNAMIC_COPY);
	glBindBuffer(GL_ARRAY_BUFFER, mGlBuffer);
	cudaGraphicsGLRegisterBuffer(&mCudaResource, mGlBuffer, cudaGraphicsRegisterFlagsNone);
}

template <typename T>
CudaGlArrayBuffer<T>::~CudaGlArrayBuffer()
{
	glDeleteBuffers(1, &mGlBuffer);
}

template <typename T>
T* CudaGlArrayBuffer<T>::mapCuda(cudaStream_t cudaStream)
{
	if(mCudaMapped) throw CudaGlArrayBufferException("can't map to cuda");

	auto error = cudaGraphicsMapResources(1, &mCudaResource, cudaStream);
	if(error != CUDA_SUCCESS) throw CudaGlArrayBufferException(cudaGetErrorString(error));
	
	size_t size;
	T* mappedPointer;
	error = cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mappedPointer), &size, mCudaResource);
	if(error != CUDA_SUCCESS) throw CudaGlArrayBufferException(cudaGetErrorString(error));

	mCudaMapped = true;
	return mappedPointer;
}

template <typename T>
void CudaGlArrayBuffer<T>::unmapCuda(cudaStream_t cudaStream)
{
	if(!mCudaMapped) throw CudaGlArrayBufferException("can't unmap from cuda");

	auto error = cudaGraphicsUnmapResources(1, &mCudaResource, cudaStream);
	if(error != CUDA_SUCCESS) throw CudaGlArrayBufferException(cudaGetErrorString(error));

	mCudaMapped = false;
}

template <typename T>
void CudaGlArrayBuffer<T>::bindGl()
{
	if(mCudaMapped) throw CudaGlArrayBufferException("can't map to gl");

	glBindBuffer(GL_ARRAY_BUFFER, mGlBuffer);
}


void DEFINE_VALID_TEMPLATES(CudaGlArrayBuffer<float>)
{
	throw "do not call me";
	CudaGlArrayBuffer<float> f(0,0);
	CudaGlArrayBuffer<double> d(0, 0);
	CudaGlArrayBuffer<int> i(0, 0);
	CudaGlArrayBuffer<unsigned int> ui(0, 0);
	CudaGlArrayBuffer<short> s(0, 0);
	CudaGlArrayBuffer<unsigned short> us(0, 0);
	CudaGlArrayBuffer<long> l(0, 0);
	CudaGlArrayBuffer<unsigned long> ul(0, 0);
}
