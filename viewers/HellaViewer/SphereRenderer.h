#pragma once
#include "Renderer.h"
#include "SubdivisionSphere.h"
#include "Texture.h"
#include "Parameters.h"


class SphereRenderer :
	public Renderer
{
public:
	enum class RenderMode
	{
		POSITION, ROTATION, LIGHTFIELD, LIGHTFIELD_SLICE
	};
	SphereRenderer(const Parameters& prmIn);
	~SphereRenderer();
	void initialize() override;
	void update(double timestep) override;
	void render(const RenderData& renderData) override;
	void selectRenderMode(RenderMode mode);
	void increaseLevel();
	void decreaseLevel();
	void highlightFaces(const std::vector<unsigned int>& faceIds);
	void highlightVertices(const std::vector<unsigned>& vertexIds);
private:
	void renderLightfield(const RenderData & renderData) const;
	void renderRotationSpheres(const RenderData & renderData);
	void renderPositionSphere(const RenderData & renderData);

	void cleanupGlBuffers();
	void setupGlBuffersForLevel(unsigned short level);
	void generateLightfield();
	std::shared_ptr<LightField::SubdivisionSphere> mSphereData;
	std::unique_ptr<Generator::LightFieldСontainer> mLightfieldContainer;
	GLuint mVertexBuffer, mIndexBuffer, mLightfieldSliceBuffer, mCompleteLightfieldBuffer, mVertexArrayObject;
	GLuint mGlProgram, mLightfieldGlProgram, mHighlightFacesGlProgram, mHighlightVerticesGlProgram, mRotationSphereGlProgram;
	unsigned int mFacesCount, mCurrentLevel;
	std::vector<unsigned int> mHighlightFaces, mHighlightVertices;
	std::unique_ptr<Texture> mDebugTexture;
	RenderMode mRenderMode;
	const Parameters& prm;
};
