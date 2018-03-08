#include "stdafx.h"
#include "SphereRenderer.h"
#include "GLUtility.h"
#include "ShaderManager.h"
#include <climits>
using namespace std; // too many std calls :D

SphereRenderer::SphereRenderer(unsigned int levelCount) :
	mSphereData(make_unique<SubdivisionShpere::SubdivisionSphere>(levelCount)), mFacesCount(0), mCurrentLevel(0)
{
}


SphereRenderer::~SphereRenderer()
{
	cleanupGlBuffers();
	glDeleteProgram(mGlProgram);
}

void SphereRenderer::initialize()
{
	mGlProgram = ShaderManager::instance().from("shader/sphereRenderer.vert", "shader/sphereRenderer.geom", "shader/sphereRenderer.frag");
	mHighlightFacesGlProgram = ShaderManager::instance().from("shader/sphereRenderer.vert", "shader/sphereRenderer.geom", "shader/sphereRendererHighlightFaces.frag");
	mHighlightVerticesGlProgram = ShaderManager::instance().from("shader/sphereRendererHighlightVertices.vert", "shader/sphereRendererHighlightVertices.frag");
	mDebugTexture = make_unique<Texture>("E:\\crohmann\\tmp\\world_texture.jpg");
	mDebugTexture->asyncTransferToGPU(std::chrono::duration<int>::max());
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::update(double timestep)
{
}

void SphereRenderer::render(const RenderData& renderData)
{
	const auto viewspaceLightPosition = glm::vec3(0); // lul

	// -------- drawing base sphere frontface --------
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunc(GL_ONE, GL_SRC_ALPHA);

	glUseProgram(mGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mGlProgram, "mvp", renderData.viewProjectionMatrix);
	GLUtility::setUniform(mGlProgram, "viewMatrix", renderData.viewMatrix);
	GLUtility::setUniform(mGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	GLUtility::setUniform(mGlProgram, "alphaMult", 0.95f);
	GLUtility::setUniform(mGlProgram, "alphaOut", 0.05f);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));


	// ------- drawing debug vertices --------
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glUseProgram(mHighlightVerticesGlProgram);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "mvp", renderData.viewProjectionMatrix);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "viewMatrix", renderData.viewMatrix);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "pointSize", 80.0f / (mCurrentLevel + 1.0f));
	for(auto i : mHighlightVertices)
	{
		const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);
		if(i < sphereLevel.numberOfVertices)
		{
			glDrawArrays(GL_POINTS, i, 1u);
		}
	}

	// ------- drawing debug faces -------
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);

	glUseProgram(mHighlightFacesGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mHighlightFacesGlProgram, "mvp", renderData.viewProjectionMatrix);
	GLUtility::setUniform(mHighlightFacesGlProgram, "viewMatrix", renderData.viewMatrix);
	GLUtility::setUniform(mHighlightFacesGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	for(auto i : mHighlightFaces)
	{
		const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);
		if(i < sphereLevel.numberOfFaces)
		{
			glDrawElements(GL_TRIANGLES, 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(sizeof(unsigned int) * 3u * i));
		}
	}

	// -------- drawing base sphere backface ---------
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);

	glUseProgram(mGlProgram);
	GLUtility::setUniform(mGlProgram, "alphaMult", 1.0f);
	GLUtility::setUniform(mGlProgram, "alphaOut", 1.0f);

	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
	glDisable(GL_BLEND);
}

void SphereRenderer::increaseLevel()
{
	++mCurrentLevel;
	cleanupGlBuffers();
	if(mCurrentLevel >= mSphereData->getNumberOfLevels()) mCurrentLevel = mSphereData->getNumberOfLevels() - 1u;
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::decreaseLevel()
{
	--mCurrentLevel;
	cleanupGlBuffers();
	if(mCurrentLevel >= mSphereData->getNumberOfLevels()) mCurrentLevel = 0;
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::highlightFaces(const std::vector<unsigned>& faceIds)
{
	mHighlightFaces = faceIds;
}

void SphereRenderer::highlightVertices(const std::vector<unsigned>& vertexIds)
{
	mHighlightVertices = vertexIds;
}

void SphereRenderer::cleanupGlBuffers()
{
	glDeleteBuffers(1, &mVertexBuffer);
	glDeleteBuffers(1, &mIndexBuffer);
	glDeleteVertexArrays(1, &mVertexArrayObject);
}

void SphereRenderer::setupGlBuffersForLevel(unsigned short level)
{
	auto sphereLevel = mSphereData->getLevel(level);
	mFacesCount = sphereLevel.numberOfFaces;

	{
		auto indices = make_unique<vector<unsigned int>>(); // shorts are not enough :o 
		indices->reserve(sphereLevel.numberOfFaces * 3);
		for(auto faceIterator = sphereLevel.faces; faceIterator != (sphereLevel.faces + sphereLevel.numberOfFaces); ++faceIterator)
		{
			indices->push_back(faceIterator->vertA);
			indices->push_back(faceIterator->vertB);
			indices->push_back(faceIterator->vertC);
		}
		mIndexBuffer = GLUtility::generateBuffer(GL_ELEMENT_ARRAY_BUFFER, *indices, GL_STATIC_DRAW);
		// TODO direct transfer using bufferSubData
	}

	{
		auto vertices = make_unique<vector<glm::vec3>>();
		vertices->reserve(sphereLevel.numberOfVertices);
		for(auto vertexIterator = sphereLevel.vertices; vertexIterator != (sphereLevel.vertices + sphereLevel.numberOfVertices); ++vertexIterator)
		{
			vertices->push_back(vertexIterator->position);
		}
		mVertexBuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, *vertices, GL_STATIC_DRAW);
		// TODO direct transfer using bufferSubData
		// TODO maybe even transfer whole Sphere::Vertex and define appropriate VertexAttribPointers?
	}


	glGenVertexArrays(1, &mVertexArrayObject);
	glBindVertexArray(mVertexArrayObject);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
	auto vertexBufferLocation = glGetAttribLocation(mGlProgram, "vertex");
	glVertexAttribPointer(vertexBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertexBufferLocation);

	glBindVertexArray(0);
}
