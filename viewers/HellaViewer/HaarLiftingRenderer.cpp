#include "stdafx.h"
#include "HaarLiftingRenderer.h"
#include "GLUtility.h"
#include "CudaHaarLifting.cpp"
#include "ShaderManager.h"


HaarLiftingRenderer::HaarLiftingRenderer(const unsigned int n): mLifter(n), mXAxis(0), mYAxis(0), mGlProgram(0), mVao(0)
{
}


void HaarLiftingRenderer::initialize()
{
	std::vector<float> xCoords;
	for(auto i = 0u; i < mLifter.size(); ++i)
	{
		xCoords.push_back(static_cast<float>(i));
	}
	mXAxis = GLUtility::generateBuffer(GL_ARRAY_BUFFER, xCoords, GL_STATIC_DRAW);

	mGlProgram = ShaderManager::instance().from("shader/haarLifting.vert", "shader/haarLifting.frag");

	mLifter.generateData([](auto index) { return static_cast<float>(index * 2 + 3); });
	mLifter.uploadData();
	mYAxis = mLifter.uploadDataGl();
	mLifter.mapGl();
	//mLifter.calculateCuda();
	//mLifter.unmapGl();

	float scale = 1.f / static_cast<float>(mLifter.size());
	glUseProgram(mGlProgram);
	glUniform1fv(glGetUniformLocation(mGlProgram, "scale"), 1, &scale);
	glGenVertexArrays(1, &mVao);
	glBindVertexArray(mVao);
	{
		glBindBuffer(GL_ARRAY_BUFFER, mXAxis);
		const auto xAxisUniformLocation = glGetAttribLocation(mGlProgram, "xCoord");
		glEnableVertexAttribArray(xAxisUniformLocation);
		glVertexAttribPointer(xAxisUniformLocation, 1, GL_FLOAT, GL_FALSE, 0, 0);

		const auto haarBufferSpec = mLifter.requiredGlBufferSpec();
		glBindBuffer(GL_ARRAY_BUFFER, mYAxis);
		const auto yAxisUniformLocation = glGetAttribLocation(mGlProgram, "yCoord");
		glVertexAttribPointer(yAxisUniformLocation, 1, GL_FLOAT, GL_FALSE, 0, haarBufferSpec.mByteOffset);
		glEnableVertexAttribArray(yAxisUniformLocation);
	}
	glBindVertexArray(0);


}

void HaarLiftingRenderer::update(double timestep)
{
}

void HaarLiftingRenderer::render(const glm::mat4x4 & viewProjection, const glm::vec3 & eyePosition)
{
	glUseProgram(mGlProgram);

	glBindVertexArray(mVao);

	glUniformMatrix4fv(glGetUniformLocation(mGlProgram, "mvp"), 1, GL_FALSE, &viewProjection[0][0]);

	glEnable(GL_LINE_SMOOTH);
	glLineWidth(1.f);
	glDrawArrays(GL_LINE_STRIP, 1, mLifter.size()-1);

	glUseProgram(0);
}

void HaarLiftingRenderer::calculate()
{
	mCudaThread = std::move(mLifter.calculateCuda());
}
