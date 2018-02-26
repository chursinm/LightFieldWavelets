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
	CudaHaarLifting<int, int2> mLifter;
};
