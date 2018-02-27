#include "stdafx.h"
#include "CheckerboardRenderer.h"
#include "ShaderManager.h"
#include "Blit.h"


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

void CheckerboardRenderer::render(const glm::mat4x4 & viewProjection, const glm::vec3 & eyePosition)
{
	glUseProgram(mGlProgram);
	glDepthMask(GL_FALSE);

	auto invViewProjection = glm::inverse(viewProjection);
	glUniformMatrix4fv(glGetUniformLocation(mGlProgram, "vp"), 1, GL_FALSE, &viewProjection[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(mGlProgram, "ivp"), 1, GL_FALSE, &invViewProjection[0][0]);
	glUniform3fv(glGetUniformLocation(mGlProgram, "eyepos"), 1, &eyePosition[0]);

	Blit::instance().render();

	glDepthMask(GL_TRUE);
	glUseProgram(0);
}
