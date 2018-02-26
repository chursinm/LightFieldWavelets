#include "stdafx.h"
#include "HaarLiftingRenderer.h"
#include "GLUtility.h"
#include "CudaHaarLifting.cpp"


HaarLiftingRenderer::HaarLiftingRenderer(unsigned int n): mLifter(n)
{
}


void HaarLiftingRenderer::initialize()
{
	//CudaHaarLifting chl(29u); // max, 2^29 ~ eine halbe Milliarde -> 2GB raw input, 5GB worksize
	mLifter.generateData();
	mLifter.uploadData();
	//m_lifter.calculateReference();
	//m_lifter.calculateCuda(); // split, predict, update
	//m_lifter.downloadData();
	//std::cout << "Haar Lifting check " << m_lifter.checkResult();

	std::vector<float> xCoords;
	for(auto i = 0u; i < mLifter.size(); ++i)
	{
		xCoords.push_back(static_cast<float>(i));
	}
	auto xBuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, xCoords, GL_STATIC_DRAW);
}

void HaarLiftingRenderer::update(double timestep)
{
}

void HaarLiftingRenderer::render(const glm::mat4x4 & viewProjection, const glm::vec3 & eyePosition)
{
}

void HaarLiftingRenderer::calculate()
{
	mLifter.calculateCuda();
}
