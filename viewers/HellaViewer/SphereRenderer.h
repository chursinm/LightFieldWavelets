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
	void increaseLevel();
	void decreaseLevel();
private:
	void cleanupGlBuffers();
	void setupGlBuffersForLevel(unsigned short level);
	std::unique_ptr<SubdivisionShpere::SubdivisionSphere> mSphereData;
	GLuint mVertexBuffer, mIndexBuffer, mVertexArrayObject, mGlProgram;
	unsigned int mFacesCount, mCurrentLevel;
};
