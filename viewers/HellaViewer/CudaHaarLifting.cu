#include "cuda_runtime.h"
#include "device_launch_parameters.h"

template <typename T, typename Vector2T>
__global__ void splitPredictUpdate(Vector2T * input, T * oddOut, T * evenOut, T threadcount)
{
	auto gid = blockIdx.x * blockDim.x + threadIdx.x;
	if(gid >= threadcount) return;

	// split
	auto evenOdd = input[gid];
	auto evenI = evenOdd.x;
	auto oddI = evenOdd.y;

	// predict
	oddI -= evenI;

	// update
	//evenI += (oddI >> 1);
	evenI += (oddI / 2);

	// store result
	oddOut[gid] = oddI;
	evenOut[gid] = evenI;
};

template <typename T, typename Vector2T>
void callSplitPredictUpdate(dim3 grid, dim3 threads, cudaStream_t stream, Vector2T * input, T * oddOut, T * evenOut, T threadcount)
{
	splitPredictUpdate<<<grid,threads,0,stream>>>(input, oddOut, evenOut, threadcount);
};

void DEFINE_VALID_TEMPLATES()
{
	throw "do not call me";
	callSplitPredictUpdate<float, float2>(0, 0, 0, 0, 0, 0, 0);
	callSplitPredictUpdate<double, double2>(0, 0, 0, 0, 0, 0, 0);
}