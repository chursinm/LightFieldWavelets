// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CudaHaarLifting.h"
#include "CameraArrayRenderer.h"
#include "CheckerboardRenderer.h"
#include "HellaViewer.h"

int main(int argc, char** argv)
{
	CudaPropertyViewer cpv;
	cpv.print();

	bool quit = false;
	auto t = std::thread([] {
		CudaHaarLifting chl(29u); // max, 2^29 ~ eine halbe Milliarde -> 2GB raw input, 5GB worksize
		chl.generateData();
		chl.uploadData();
		chl.calculateReference();
		chl.calculateCuda(); // split, predict, update
		chl.downloadData();
		std::cout << "Haar Lifting check " << chl.checkResult();
	});

	RenderContext context;
	CheckerboardRenderer cbr;
	CameraArrayRenderer car;
	context.attachRenderer(cbr);
	context.attachRenderer(car);
	context.onKeyPress([&car](auto keymod, auto keycode) { car.handleInput(keymod, keycode); });

	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}