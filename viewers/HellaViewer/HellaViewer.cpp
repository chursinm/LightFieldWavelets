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



void play_with_haar_lifting();

int main(int argc, char** argv)
{
	CudaPropertyViewer cpv;
	cpv.print();

	bool quit = false;
	
	RenderContext context(true, true);

	CheckerboardRenderer cbr;
	context.attachRenderer(cbr);

	SphereRenderer sr(6u);
	//sr.highlightFaces({ 1u,3u,5u,10u });
	//sr.highlightVertices({ 1u,3u,5u,10u });
	context.attachRenderer(sr);
	context.onKeyPress([&sr](auto keymod, auto keycode)
	{
		if(keycode == SDLK_LESS) sr.selectRenderMode(SphereRenderer::RenderMode::POSITION);
		if(keycode == SDLK_y) sr.selectRenderMode(SphereRenderer::RenderMode::ROTATION);
		if(keycode == SDLK_x) sr.selectRenderMode(SphereRenderer::RenderMode::LIGHTFIELD);
		if(keycode == SDLK_c) sr.selectRenderMode(SphereRenderer::RenderMode::LIGHTFIELD_SLICE);
	});
	context.onKeyPress([&sr](auto keymod, auto keycode) { if(keycode == SDLK_b) sr.increaseLevel(); else if(keycode == SDLK_v) sr.decreaseLevel(); });

	if(!context.initialize()) { quit = true; }

	while(!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

	return 0;
}


void play_with_haar_lifting()
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
}