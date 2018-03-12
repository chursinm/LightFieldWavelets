#include "stdafx.h"
#include "SphereRenderer.h"
#include "GLUtility.h"
#include "ShaderManager.h"
#include <climits>
#include <glm/gtc/matrix_transform.inl>
#include "LightfieldLevel.h"
#include "ColoredPlaneSampler.h"
#include "SetSampler.h"
#include "CheckerboardSampler.h"
using namespace std; // too many std calls :D
#define DEBUG_ROTATION_SPHERES 0
#define RENDER_LIGHTFIELD 1

SphereRenderer::SphereRenderer(unsigned int levelCount) :
	mSphereData(make_shared<SubdivisionShpere::SubdivisionSphere>(levelCount)), mFacesCount(0), mCurrentLevel(0)
{
	generateLightfieldForLevel(mCurrentLevel);
}


SphereRenderer::~SphereRenderer()
{
	cleanupGlBuffers();
	glDeleteProgram(mGlProgram);
	glDeleteProgram(mLightfieldGlProgram);
	glDeleteProgram(mHighlightFacesGlProgram);
	glDeleteProgram(mHighlightVerticesGlProgram);
}

void SphereRenderer::initialize()
{
	mGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRenderer.vert", "shader/sphereRenderer/sphereRenderer.geom", "shader/sphereRenderer/sphereRenderer.frag");
	mLightfieldGlProgram = ShaderManager::instance().from("shader/sphereRenderer/lightfieldRenderer.vert", "shader/sphereRenderer/lightfieldRenderer.frag");
	mHighlightFacesGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRenderer.vert", "shader/sphereRenderer/sphereRenderer.geom", "shader/sphereRenderer/sphereRendererHighlightFaces.frag");
	mHighlightVerticesGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRendererHighlightVertices.vert", "shader/sphereRenderer/sphereRendererHighlightVertices.frag");
	mDebugTexture = make_unique<Texture>("E:\\crohmann\\tmp\\world_texture.jpg");
	mDebugTexture->asyncTransferToGPU(std::chrono::duration<int>::max());
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::update(double timestep)
{
}

void SphereRenderer::render(const RenderData& renderData)
{


#if DEBUG_ROTATION_SPHERES
	const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);
	for(auto vertexIterator = sphereLevel.vertices; vertexIterator != (sphereLevel.vertices + sphereLevel.numberOfVertices); ++vertexIterator)
	{
	auto face = sphereLevel.faces[0];
	auto posA = face.vertARef->position;
	auto posB = face.vertBRef->position;
	auto scaleFac = glm::distance(posA, posB) * 0.66666f;
#endif
	auto modelMat = glm::mat4x4(1.f);
#if DEBUG_ROTATION_SPHERES
	modelMat = glm::translate(modelMat, vertexIterator->position);
	modelMat = glm::scale(modelMat, glm::vec3(1.05f*scaleFac));
#endif

	const auto viewProjection = renderData.viewProjectionMatrix * modelMat;
	const auto viewMatrix = renderData.viewMatrix * modelMat;
	const auto viewspaceLightPosition = viewMatrix * glm::vec4(0.f, 10.f, 0.f, 1.f);


#if RENDER_LIGHTFIELD
	const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);
	const auto lfData = mLightfieldLevel->snapshot(renderData.eyePositionWorld);

	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, lfData.size() * sizeof(glm::vec3), &lfData[0]);

	glDisable(GL_CULL_FACE);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glDisable(GL_BLEND);

	glUseProgram(mLightfieldGlProgram);
	GLUtility::setUniform(mLightfieldGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mLightfieldGlProgram, "viewMatrix", viewMatrix);
	GLUtility::setUniform(mLightfieldGlProgram, "viewspaceLightPosition", viewspaceLightPosition);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
	return;
#endif

	// -------- drawing base sphere frontface --------
	glEnable(GL_CULL_FACE);
#if !DEBUG_ROTATION_SPHERES
	glCullFace(GL_FRONT);
#endif

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	//glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_SRC_ALPHA);

	glUseProgram(mGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mGlProgram, "viewMatrix", viewMatrix);
	GLUtility::setUniform(mGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	GLUtility::setUniform(mGlProgram, "alphaMult", 0.95f);
	GLUtility::setUniform(mGlProgram, "alphaOut", 0.05f);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
#if DEBUG_ROTATION_SPHERES
	continue;
#endif

	// ------- drawing debug vertices --------
	glDisable(GL_CULL_FACE);
	glDisable(GL_BLEND);
	glDisable(GL_DEPTH_TEST);

	glEnable(GL_PROGRAM_POINT_SIZE);
	glUseProgram(mHighlightVerticesGlProgram);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mHighlightVerticesGlProgram, "viewMatrix", viewMatrix);
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
	GLUtility::setUniform(mHighlightFacesGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mHighlightFacesGlProgram, "viewMatrix", viewMatrix);
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
#if DEBUG_ROTATION_SPHERES
	}
#endif
}

void SphereRenderer::increaseLevel()
{
	++mCurrentLevel;
	cleanupGlBuffers();
	if(mCurrentLevel >= mSphereData->getNumberOfLevels()) mCurrentLevel = mSphereData->getNumberOfLevels() - 1u;
	setupGlBuffersForLevel(mCurrentLevel);
	generateLightfieldForLevel(mCurrentLevel);
	std::cout << "Level: " << mCurrentLevel << " ";
}

void SphereRenderer::decreaseLevel()
{
	--mCurrentLevel;
	cleanupGlBuffers();
	if(mCurrentLevel >= mSphereData->getNumberOfLevels()) mCurrentLevel = 0;
	setupGlBuffersForLevel(mCurrentLevel);
	generateLightfieldForLevel(mCurrentLevel);
	std::cout << "Level: " << mCurrentLevel << " ";
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
	glDeleteBuffers(1, &mLightfieldBuffer);
	glDeleteVertexArrays(1, &mVertexArrayObject);
}

void SphereRenderer::setupGlBuffersForLevel(unsigned short level)
{
	auto sphereLevel = mSphereData->getLevel(level);
	mFacesCount = sphereLevel.numberOfFaces;

	// Upload Indices
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

	// Upload Vertices
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

	// Allocate lightfield space
	{
	#if RENDER_LIGHTFIELD
		mLightfieldBuffer = GLUtility::generateBuffer<glm::vec3>(GL_ARRAY_BUFFER, sphereLevel.numberOfVertices, nullptr, GL_DYNAMIC_DRAW);
	#endif
	}


	glGenVertexArrays(1, &mVertexArrayObject);
	glBindVertexArray(mVertexArrayObject);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
	const auto vertexBufferLocation = glGetAttribLocation(mGlProgram, "vertex");
	glVertexAttribPointer(vertexBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(vertexBufferLocation);
	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldBuffer);
#if RENDER_LIGHTFIELD
	const auto lightfieldBufferLocation = 1; // glGetAttribLocation(mGlProgram, "lightfield");
	glVertexAttribPointer(lightfieldBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glEnableVertexAttribArray(lightfieldBufferLocation);
#endif

	glBindVertexArray(0);
}

void SphereRenderer::generateLightfieldForLevel(unsigned short level)
{
	// some aliases for readability
	using namespace Generator::Sampler;
	using namespace glm;
	using Plane = PlaneSampler::Plane;
	using Ray = Sampler::Ray;

	// create Sampler
	//Plane plane(vec3(0, 0, -10), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0)); // mind the backface culling!
	//const auto planeSampler = make_shared<CheckerboardSampler>(0.f, vec3(1.0f, 0.f, 0.f), plane);
	Plane plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0)); // mind the backface culling!
	const auto planeSampler = make_shared<CheckerboardSampler>(0.f, vec3(0.1f), plane);

#if RENDER_LIGHTFIELD
	mLightfieldLevel = make_unique<Generator::LightfieldLevel>(mSphereData, level, *planeSampler);
#endif
}
