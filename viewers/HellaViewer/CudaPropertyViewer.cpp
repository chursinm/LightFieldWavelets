/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and
* proprietary rights in and to this software and related documentation.
* Any use, reproduction, disclosure, or distribution of this software
* and related documentation without an express license agreement from
* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA)
* associated with this source code for terms and conditions that govern
* your use of this NVIDIA software.
*
*/

#include "stdafx.h"
#include "CudaPropertyViewer.h"
#include <cuda.h>
#include <cuda_runtime_api.h>


CudaPropertyViewer::CudaPropertyViewer()
{
	int count;
	auto error = cudaGetDeviceCount(&count);
	if(error != CUDA_SUCCESS)
	{
		std::cout << "Error using Cuda: " << cudaGetErrorString(error) << std::endl;
		return;
	}
	for(int i = 0; i < count; i++)
	{
		error = cudaSetDevice(i);
		if(error != CUDA_SUCCESS)
		{
			std::cout << "Error using Cuda: " << cudaGetErrorString(error) << std::endl;
			continue;
		}
		cudaDeviceProp prop;
		error = cudaGetDeviceProperties(&prop, i);
		if(error != CUDA_SUCCESS)
		{
			std::cout << "Error using Cuda: " << cudaGetErrorString(error) << std::endl;
			continue;
		}
		props.push_back(prop);
	}
}


CudaPropertyViewer::~CudaPropertyViewer()
{
}

void CudaPropertyViewer::print()
{
	unsigned int i = 0;
	for(auto& prop : props)
	{
		printf("   --- General Information for device %d ---\n", i++);
		printf("Name:  %s\n", prop.name);
		printf("Compute capability:  %d.%d\n", prop.major, prop.minor);
		printf("Clock rate:  %d\n", prop.clockRate);
		printf("Device copy overlap:  ");
		if (prop.deviceOverlap)
			printf("Enabled\n");
		else
			printf("Disabled\n");
		printf("Kernel execution timeout :  ");
		if (prop.kernelExecTimeoutEnabled)
			printf("Enabled\n");
		else
			printf("Disabled\n");

		printf("   --- Memory Information for device %d ---\n", i);
		printf("Total global mem:  %ld\n", prop.totalGlobalMem);
		printf("Total constant Mem:  %ld\n", prop.totalConstMem);
		printf("Max mem pitch:  %ld\n", prop.memPitch);
		printf("Texture Alignment:  %ld\n", prop.textureAlignment);

		printf("   --- MP Information for device %d ---\n", i);
		printf("Multiprocessor count:  %d\n",
			prop.multiProcessorCount);
		printf("Shared mem per mp:  %ld\n", prop.sharedMemPerBlock);
		printf("Registers per mp:  %d\n", prop.regsPerBlock);
		printf("Threads in warp:  %d\n", prop.warpSize);
		printf("Max threads per block:  %d\n",
			prop.maxThreadsPerBlock);
		printf("Max thread dimensions:  (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2]);
		printf("Max grid dimensions:  (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2]);
		printf("\n");
	}

	
}
