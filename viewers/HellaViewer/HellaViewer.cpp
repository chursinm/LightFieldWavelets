// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CameraArrayRenderer.h"

__global__ void kernel(void)
{

}

int main(int argc, char** argv)
{
	kernel << <1, 1 >> > ();
	CudaPropertyViewer::print();

	bool quit = false;

	RenderContext context;
	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}
