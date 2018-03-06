#pragma once
#include "Renderer.h"
#include "SubdivisionSphere.h"
#include "Texture.h"

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
	void highlightFaces(const std::vector<unsigned int>& faceIds);
private:
	void cleanupGlBuffers();
	void setupGlBuffersForLevel(unsigned short level);
	std::unique_ptr<SubdivisionShpere::SubdivisionSphere> mSphereData;
	GLuint mVertexBuffer, mIndexBuffer, mVertexArrayObject, mGlProgram, mHighlightFacesGlProgram;
	unsigned int mFacesCount, mCurrentLevel;
	std::vector<unsigned int> mHighlightFaces, mHighlightVertices;
	std::unique_ptr<Texture> mDebugTexture;
};
