// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CameraArrayRenderer.h"
#include "CudaHaarLifting.h"

void bindCameraArrayRenderer(RenderContext& rc, CameraArrayRenderer& car);

int main(int argc, char** argv)
{
	CudaPropertyViewer cpv;
	cpv.print();

	bool quit = false;
	CudaHaarLifting chl(29u); // max, 2^29 ~ eine halbe Milliarde -> 2GB raw input, 5GB worksize
	chl.generateData();
	chl.uploadData();
	chl.calculateReference();
	chl.calculateCuda(); // split, predict, update
	chl.downloadData();
	std::cout << "Haar Lifting check " << chl.checkResult();

	RenderContext context;
	CameraArrayRenderer car;
	bindCameraArrayRenderer(context, car);

	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}


void bindCameraArrayRenderer(RenderContext& rc, CameraArrayRenderer& car)
{
	rc.postInitialize([&] { car.initialize(); });
	rc.onRenderEyeTexture([&](const EyeTextureParamaterization& params) { car.render(params.viewProjection, params.cameraPositionWorld); });
	rc.onKeyPress([&](auto keymod, auto keycode)
	{
		// TODO this is logic and shouldn't be part of the binding
		auto cameraSpeedInSceneUnitPerMS = 0.01f;
		if(keymod & (SDL_Keymod::KMOD_LSHIFT | SDL_Keymod::KMOD_RSHIFT)) cameraSpeedInSceneUnitPerMS *= 10.f;
		switch(keycode)
		{
		case SDLK_o:
			car.m_FocalPlane += cameraSpeedInSceneUnitPerMS;
			std::cout << "Focal plane: " << car.m_FocalPlane << std::endl;
			break;
		case SDLK_l:
			car.m_FocalPlane -= cameraSpeedInSceneUnitPerMS;
			std::cout << "Focal plane: " << car.m_FocalPlane << std::endl;
			break;
		}
	});
}