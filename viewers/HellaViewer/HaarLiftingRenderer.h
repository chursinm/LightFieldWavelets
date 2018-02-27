#pragma once
#include "Renderer.h"
#include "CudaHaarLifting.h"
class HaarLiftingRenderer :
	public Renderer
{
public:
	explicit HaarLiftingRenderer(unsigned int n);
	~HaarLiftingRenderer() = default;

	void initialize() override;
	void update(double timestep) override;
	void render(const glm::mat4x4 & viewProjection, const glm::vec3 & eyePosition) override;
	void calculate();

private:
	CudaHaarLifting<float, float2> mLifter;
	GLuint mXAxis, mYAxis;
	GLuint mGlProgram;
	GLuint mVao;
	std::future<void> mCudaThread;
};
