// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <chrono>
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CheckerboardRenderer.h"
#include "HellaViewer.h"
#include "CudaHaarLifting.cpp"

int main(int argc, char** argv)
{
	CudaPropertyViewer cpv;
	cpv.print();

	bool quit = false;

	RenderContext context;
	CheckerboardRenderer cbr;
	context.attachRenderer(cbr);
	//CameraArrayRenderer car;
	//context.attachRenderer(car);
	//context.onKeyPress([&car](auto keymod, auto keycode) { car.handleInput(keymod, keycode); });
	//HaarLiftingRenderer hlr(4u);
	//context.attachRenderer(hlr);
	//context.onKeyPress([&hlr](auto keymod, auto keycode) { if(keycode & SDLK_x) hlr.calculate(); });

	if(!context.initialize()) { quit = true; }


	CudaHaarLifting<float, float2> chl(14u); // max, 2^29 ~ eine halbe Milliarde -> 2GB raw input + 3GB worksize
	chl.generateData();
	chl.uploadData();
	chl.uploadDataGL();
	chl.mapGl();
	auto asynCalculate = [&chl]()
	{
		try
		{
			chl.calculateCuda(); // split, predict, update
		}
		catch(CudaHaarLiftingException e)
		{
			std::cout << e.what();
		}
		chl.calculateReference();
		return 1;
	};
	const auto futureLaunchType = std::launch::async;

	std::future<int> future;

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
		if(!future.valid()) future = std::async(futureLaunchType, asynCalculate);
		if(futureLaunchType == std::launch::deferred && future.valid()) future.get();
		if(futureLaunchType == std::launch::async && future.valid() && future.wait_for(std::chrono::duration<int>::zero()) == std::future_status::ready)
		{
			//chl.mapGl();
			chl.downloadData();
			chl.unmapGl();
			std::cout << "Haar Lifting check " << chl.checkResult();
		}
	}

	return 0;
}