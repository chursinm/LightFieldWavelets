#include "stdafx.h"
#include "CheckerboardRenderer.h"
#include "ShaderManager.h"
#include "Blit.h"
#include "GLUtility.h"


CheckerboardRenderer::CheckerboardRenderer()
{
}


CheckerboardRenderer::~CheckerboardRenderer()
{
}

void CheckerboardRenderer::initialize()
{
	// Create GL programs
	ShaderManager& psm = ShaderManager::instance();
	mGlProgram = psm.from("shader/stereo.vert", "shader/stereo.frag");
	if(mGlProgram == 0)
		throw "couldn't load shaders";
}

void CheckerboardRenderer::update(double timestep)
{
}

void CheckerboardRenderer::render(const RenderData& renderData)
{
	glUseProgram(mGlProgram);
	glDepthMask(GL_FALSE);

	auto invViewProjection = glm::inverse(renderData.viewProjectionMatrix);
	GLUtility::setUniform(mGlProgram, "vp", renderData.viewProjectionMatrix);
	GLUtility::setUniform(mGlProgram, "ivp", invViewProjection);
	GLUtility::setUniform(mGlProgram, "eyepos", renderData.eyePositionWorld);

	Blit::instance().render();

	glDepthMask(GL_TRUE);
	glUseProgram(0);
}
