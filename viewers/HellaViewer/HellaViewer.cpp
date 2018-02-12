// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CameraArrayRenderer.h"
#include "CudaHaarLifting.h"

int main(int argc, char** argv)
{
	CudaPropertyViewer::print();

	bool quit = false;

	CudaHaarLifting chl(8u);
	chl.generateData();
	chl.uploadData();
	chl.calculateCuda();
	chl.downloadData();
	RenderContext context;
	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}
