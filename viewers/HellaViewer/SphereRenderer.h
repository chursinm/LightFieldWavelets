#pragma once
#include "Renderer.h"
#include "SubdivisionSphere.h"
class SphereRenderer :
	public Renderer
{
public:
	SphereRenderer(unsigned int levelCount);
	~SphereRenderer();
	void initialize() override;
	void update(double timestep) override;
	void render(const RenderData& renderData) override;
private:
	std::unique_ptr<SubdivisionShpere::SubdivisionSphere> mSphereData;
	GLuint mVertexBuffer, mIndexBuffer, mVertexArrayObject, mGlProgram;
	unsigned int mFacesCount;
};
