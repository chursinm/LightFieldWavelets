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
	mDebugTexture = make_unique<Texture>("E:\\crohmann\\tmp\\world_texture2.jpg");
	mDebugTexture->asyncTransferToGPU(std::chrono::duration<int>::max());
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::update(double timestep)
{
}

void SphereRenderer::render(const RenderData& renderData)
{
	const auto viewspaceLightPosition = glm::vec3(0); // lul

	// -------- drawing base sphere --------
	glDisable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glUseProgram(mGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mGlProgram, "mvp", renderData.viewProjectionMatrix);
	GLUtility::setUniform(mGlProgram, "viewMatrix", renderData.viewMatrix);
	GLUtility::setUniform(mGlProgram, "viewspaceLightPosition", viewspaceLightPosition);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));

	// ------- drawing debug faces -------
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
			indices->push_back(faceIterator->vertA - 1);
			indices->push_back(faceIterator->vertB - 1);
			indices->push_back(faceIterator->vertC - 1);
		}
		mIndexBuffer = GLUtility::generateBuffer(GL_ELEMENT_ARRAY_BUFFER, *indices, GL_STATIC_DRAW);
		// TODO direct transfer using bufferSubData
	}

	{
		auto vertices = make_unique<vector<glm::vec3>>();
		vertices->reserve(sphereLevel.numberOfVertices);
		for(auto vertexIterator = sphereLevel.vertices; vertexIterator != (sphereLevel.vertices + sphereLevel.numberOfVertices); ++vertexIterator)
		{
			vertices->push_back(glm::vec3(vertexIterator->x, vertexIterator->y, vertexIterator->z));
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
