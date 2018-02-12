// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "cuda/CudaPropertyViewer.h"
#include "CameraArrayRenderer.h"
#include "cuda/CudaHaarLifting.h"

int main(int argc, char** argv)
{
	CudaPropertyViewer::print();

	bool quit = false;

	CudaHaarLifting chl(24u);
	chl.generateData();
	chl.uploadData();
	chl.calculateReference();
	chl.calculateCuda();
	chl.downloadData();
	if(!chl.checkResult()) throw "chl failed";
	RenderContext context;
	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}
