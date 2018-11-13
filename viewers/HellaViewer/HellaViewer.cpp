// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"
#include "CudaPropertyViewer.h"
#include "CheckerboardRenderer.h"
#include "HellaViewer.h"
#include "CudaHaarLifting.cpp"
#include "HaarLiftingRenderer.h"
#include "SphereRenderer.h"
#include "CameraArrayRenderer.h"
#include "Parameters.h"




//void play_with_haar_lifting();

int main(int argc, char** argv)
{
	
	
	const Parameters prm(argc, argv);
	//CudaPropertyViewer cpv;
	//cpv.print();

	bool quit = false;
	std::shared_ptr <RenderContext> context;

	if (prm.vrMode)
	{
		context = std::make_shared<RenderContext>(true, true);
	}
	else
	{
		context = std::make_shared<RenderContext>(false, false);
	}
	

	CheckerboardRenderer cbr;
	context->attachRenderer(cbr);

	SphereRenderer sr(prm);
	//sr.highlightFaces({ 1u,3u,5u,10u });
	//sr.highlightVertices({ 1u,3u,5u,10u });
	context->attachRenderer(sr);
	context->onKeyPress([&sr](auto keymod, auto keycode)
	{
		if(keycode == SDLK_LESS) sr.selectRenderMode(SphereRenderer::RenderMode::POSITION);
		if(keycode == SDLK_y) sr.selectRenderMode(SphereRenderer::RenderMode::ROTATION);
		if(keycode == SDLK_x) sr.selectRenderMode(SphereRenderer::RenderMode::LIGHTFIELD);
		if(keycode == SDLK_c) sr.selectRenderMode(SphereRenderer::RenderMode::LIGHTFIELD_SLICE);
	});

	context->onKeyPress([&sr](auto keymod, auto keycode) { if(keycode == SDLK_b) sr.increaseLevel(); else if(keycode == SDLK_v) sr.decreaseLevel(); });
	sr.selectRenderMode(SphereRenderer::RenderMode::LIGHTFIELD_SLICE);
	if(!context->initialize()) { quit = true; }
	for (int i = 1; i < prm.numberOfLevels ; i++) { sr.increaseLevel(); }
	while(!quit)
	{
		quit = context->handleSDL();
		context->render();
	}

	return 0;
}


/*void play_with_haar_lifting()
{
	CudaHaarLifting<double, double2> chl(8u); // max, 2^29 ~ eine halbe Milliarde -> 2GB raw input + 3GB worksize
	chl.generateData([](auto index) { return static_cast<float>(index * 2 + 3); });
	chl.uploadData();
	chl.calculateReference();
	//chl.uploadDataGl();
	//chl.mapGl();
	auto future = chl.calculateCuda();
	if(future.valid())
	{
		future.wait();
		chl.downloadData();
		std::cout << "Haar Lifting check " << chl.checkResult();
	}
}*/