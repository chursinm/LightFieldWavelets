#pragma once
#include <cuda_gl_interop.h>

class CudaGlArrayBufferException : public std::exception
{
public:
	explicit CudaGlArrayBufferException(const std::string& fault) : mMessage(fault) {}
	char const * what() const override
	{
		return mMessage.c_str();
	}
private:
	std::string mMessage;
};


/*
 * Utility Wrapper to manage gl / cuda memory interop properly.
 * 
 * This class seems to be useless. Reason:
 * According to the specification we need to unmap glBuffers from Cuda, before we use them in gl. (see http://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__INTEROP.html#group__CUDART__INTEROP_1gad8fbe74d02adefb8e7efb4971ee6322 )
 * However, according to the Nvidia samples and some experimentation this seems to be obsolete. (see https://github.com/nvpro-samples/gl_cuda_interop_pingpong_st )
 * Fun fact: we can even run cuda totally independent from gl, if we manage the buffer sync. This is not according to the specs, afaik.
 * Besides these inconsistencies, binding a single resource a time opengl is not the way to go (!), rendering this class even more obsolete.
 *
 * GL properties: GL_ARRAY_BUFFER, GL_DYNAMIC_DRAW
 * 
 */
template <typename T>
class CudaGlArrayBuffer
{
public:

	explicit CudaGlArrayBuffer(unsigned int elements, T * data);
	~CudaGlArrayBuffer();

	T* mapCuda(cudaStream_t cudaStream = nullptr);
	void unmapCuda(cudaStream_t cudaStream = nullptr);
	void bindGl();
	void unbindGl();
private:
	GLuint mGlBuffer;
	cudaGraphicsResource_t mCudaResource;
	bool mCudaMapped; // in opengl we don't actually unbind single resource, so we can't have an mGlMapped.
};

