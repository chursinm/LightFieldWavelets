#include "stdafx.h"
#include "CameraArrayRenderer.h"

#define DEMO_FILE "E:/crohmann/old_input/stanford_chess_lightfield/preview/chess.xml"

CameraArrayRenderer::CameraArrayRenderer()
{
	m_CameraArray = CameraArrayParser::parse(DEMO_FILE);
}


CameraArrayRenderer::~CameraArrayRenderer()
{
}

void CameraArrayRenderer::initialize()
{
	for(auto& cam : m_CameraArray.cameras)
	{
		// For now just transfer them all blocking
		cam->tex->asyncTransferToGPU(std::chrono::duration<int>::max());
	}
}


#undef DEMO_FILE