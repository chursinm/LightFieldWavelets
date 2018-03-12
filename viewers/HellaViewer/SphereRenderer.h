#pragma once
#include "Renderer.h"
#include "SubdivisionSphere.h"
#include "Texture.h"
#include "LightfieldLevel.h"

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
	void highlightVertices(const std::vector<unsigned>& vertexIds);
private:
	void cleanupGlBuffers();
	void setupGlBuffersForLevel(unsigned short level);
	void generateLightfieldForLevel(unsigned short level);
	std::shared_ptr<SubdivisionShpere::SubdivisionSphere> mSphereData;
	std::unique_ptr<Generator::LightfieldLevel> mLightfieldLevel;
	GLuint mVertexBuffer, mIndexBuffer, mLightfieldBuffer, mVertexArrayObject;
	GLuint mGlProgram, mLightfieldGlProgram, mHighlightFacesGlProgram, mHighlightVerticesGlProgram;
	unsigned int mFacesCount, mCurrentLevel;
	std::vector<unsigned int> mHighlightFaces, mHighlightVertices;
	std::unique_ptr<Texture> mDebugTexture;
};
