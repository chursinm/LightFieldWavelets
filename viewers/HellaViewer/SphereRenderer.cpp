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
#include "TexturedPlaneSampler.h"
using namespace std; // too many std calls :D
#define DEBUG_ROTATION_SPHERES 0
#define RENDER_LIGHTFIELD 1

SphereRenderer::SphereRenderer(unsigned int levelCount) :
	mSphereData(make_shared<SubdivisionShpere::SubdivisionSphere>(levelCount)), mFacesCount(0), mCurrentLevel(1), mRenderMode(RenderMode::POSITION)
{
	generateLightfield(levelCount);
}


SphereRenderer::~SphereRenderer()
{
	cleanupGlBuffers();
	glDeleteProgram(mGlProgram);
	glDeleteProgram(mRotationSphereGlProgram);
	glDeleteProgram(mLightfieldGlProgram);
	glDeleteProgram(mHighlightFacesGlProgram);
	glDeleteProgram(mHighlightVerticesGlProgram);
}

void SphereRenderer::initialize()
{
	mGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRenderer.vert", "shader/sphereRenderer/sphereRenderer.geom", "shader/sphereRenderer/sphereRenderer.frag");
	mRotationSphereGlProgram = ShaderManager::instance().from("shader/sphereRenderer/rotationSphere.vert", "shader/sphereRenderer/rotationSphere.frag");
	mLightfieldGlProgram = ShaderManager::instance().from("shader/sphereRenderer/lightfieldRenderer.vert", "shader/sphereRenderer/lightfieldRenderer.frag");
	mHighlightFacesGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRenderer.vert", "shader/sphereRenderer/sphereRenderer.geom", "shader/sphereRenderer/sphereRendererHighlightFaces.frag");
	mHighlightVerticesGlProgram = ShaderManager::instance().from("shader/sphereRenderer/sphereRendererHighlightVertices.vert", "shader/sphereRenderer/sphereRendererHighlightVertices.frag");
	mDebugTexture = make_unique<Texture>("E:\\crohmann\\Unordered\\tmp\\textures\\world_texture.asd");
	mDebugTexture->asyncTransferToGPU(std::chrono::duration<int>::max());
	setupGlBuffersForLevel(mCurrentLevel);
}

void SphereRenderer::update(double timestep)
{
}

void SphereRenderer::renderLightfield(const RenderData& renderData) const
{
	auto modelMat = glm::mat4x4(1.f);
	const auto viewProjection = renderData.viewProjectionMatrix * modelMat;
	const auto viewMatrix = renderData.viewMatrix * modelMat;
	const auto viewspaceLightPosition = viewMatrix * glm::vec4(0.f, 10.f, 0.f, 1.f);
	
	const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);
	const auto lfData = mLightfield->level(mCurrentLevel).snapshot(renderData.eyePositionWorld);

	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldBuffer);
	glBufferSubData(GL_ARRAY_BUFFER, 0, lfData.size() * sizeof(glm::vec3), lfData.data());

	// -------- drawing base sphere backface --------
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);
	//glDisable(GL_CULL_FACE);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glDisable(GL_BLEND);

	glUseProgram(mLightfieldGlProgram);
	GLUtility::setUniform(mLightfieldGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mLightfieldGlProgram, "viewMatrix", viewMatrix);
	GLUtility::setUniform(mLightfieldGlProgram, "viewspaceLightPosition", viewspaceLightPosition);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
	
	glCullFace(GL_BACK); // shouldn't happen here..
	return;
	// -------- drawing base sphere frontface ---------
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	glUseProgram(mLightfieldGlProgram);

	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
	glDisable(GL_BLEND);
}

void SphereRenderer::renderRotationSpheres(const RenderData& renderData)
{
	const auto sphereLevel = mSphereData->getLevel(mCurrentLevel);

	// TODO do more stuff just once in here

	// calculate world scale
	auto face = sphereLevel.faces[0];
	auto posA = face.vertARef->position;
	auto posB = face.vertBRef->position;
	const auto scaleBias = 0.7f;
	auto scaleFac = glm::distance(posA, posB) * scaleBias;
	
	// mvp, mv, lightpos
	const auto modelMat = glm::mat4x4(1.f);
	const auto viewProjection = renderData.viewProjectionMatrix * modelMat;
	const auto viewMatrix = renderData.viewMatrix * modelMat;
	const auto viewspaceLightPosition = viewMatrix * glm::vec4(0.f, 10.f, 0.f, 1.f);

	// opengl
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glDisable(GL_BLEND);

	glUseProgram(mRotationSphereGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mRotationSphereGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mRotationSphereGlProgram, "viewMatrix", viewMatrix);
	GLUtility::setUniform(mRotationSphereGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	GLUtility::setUniform(mRotationSphereGlProgram, "scaleFac", scaleFac);
	GLUtility::setUniform(mRotationSphereGlProgram, "vertexCount", mSphereData->getLevel(mCurrentLevel).numberOfVertices);

	glBindVertexArray(mVertexArrayObject);

	// lightfield
	const auto lfData = mLightfield->level(mCurrentLevel).rawData();
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mCompleteLightfieldBuffer);
	//glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, lfData->size() * sizeof(glm::vec3), lfData->data());
	std::vector<glm::vec4> asd;
	for(auto a : *lfData) asd.push_back(glm::vec4(a.rgb,0.0f));
	glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, asd.size() * sizeof(glm::vec4), &asd[0]);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 3, mCompleteLightfieldBuffer);
	glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT);
	glDrawElementsInstanced(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0), sphereLevel.numberOfVertices);
}

void SphereRenderer::renderPositionSphere(const RenderData& renderData)
{
	auto modelMat = glm::mat4x4(1.f);
	const auto viewProjection = renderData.viewProjectionMatrix * modelMat;
	const auto viewMatrix = renderData.viewMatrix * modelMat;
	const auto viewspaceLightPosition = viewMatrix * glm::vec4(0.f, 10.f, 0.f, 1.f);

	// -------- drawing base sphere frontface --------
	glEnable(GL_CULL_FACE);
	glCullFace(GL_FRONT);

	glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);

	glEnable(GL_BLEND);
	glBlendFunc(GL_ONE, GL_SRC_ALPHA);

	glUseProgram(mGlProgram);
	mDebugTexture->bind(0u);
	GLUtility::setUniform(mGlProgram, "mvp", viewProjection);
	GLUtility::setUniform(mGlProgram, "viewMatrix", viewMatrix);
	GLUtility::setUniform(mGlProgram, "viewspaceLightPosition", viewspaceLightPosition);
	GLUtility::setUniform(mGlProgram, "alphaMult", 0.95f);
	GLUtility::setUniform(mGlProgram, "alphaOut", 0.05f);
	GLUtility::setUniform(mGlProgram, "renderEdges", false);

	glBindVertexArray(mVertexArrayObject);
	glDrawElements(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0));
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
}

void SphereRenderer::render(const RenderData& renderData)
{
	if(mRenderMode == RenderMode::POSITION) renderPositionSphere(renderData);
	if(mRenderMode == RenderMode::ROTATION) renderRotationSpheres(renderData);
	if(mRenderMode == RenderMode::LIGHTFIELD) renderLightfield(renderData);
}

void SphereRenderer::selectRenderMode(RenderMode mode)
{
	mRenderMode = mode;
}

void SphereRenderer::increaseLevel()
{
	if(mCurrentLevel >= mSphereData->getNumberOfLevels() - 1u)
	{
		return;
	}
	cleanupGlBuffers();
	++mCurrentLevel;
	setupGlBuffersForLevel(mCurrentLevel);
	std::cout << "Level: " << mCurrentLevel << " ";
}

void SphereRenderer::decreaseLevel()
{
	if(mCurrentLevel <= 1u)
	{
		return;
	}
	cleanupGlBuffers();
	--mCurrentLevel;
	setupGlBuffersForLevel(mCurrentLevel);
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
	glDeleteBuffers(1, &mCompleteLightfieldBuffer);
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
	}

	// Allocate lightfield space
	{
		mLightfieldBuffer = GLUtility::generateBuffer<glm::vec3>(GL_ARRAY_BUFFER, sphereLevel.numberOfVertices, nullptr, GL_DYNAMIC_DRAW);
		mCompleteLightfieldBuffer = GLUtility::generateBuffer<glm::vec4>(GL_SHADER_STORAGE_BUFFER, mLightfield->level(level).rawData()->size(), nullptr, GL_DYNAMIC_DRAW);
	}

	// Create VAO
	glGenVertexArrays(1, &mVertexArrayObject);
	glBindVertexArray(mVertexArrayObject);

	// vao - indices
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, mIndexBuffer);
	
	// vao - positions
	const auto vertexBufferLocation = 0; // glGetAttribLocation(mGlProgram, "vertex");
	glEnableVertexAttribArray(vertexBufferLocation);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
	glVertexAttribPointer(vertexBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	
	// vao - lf slice data
	const auto lightfieldBufferLocation = 1; // glGetAttribLocation(mLightfieldGlProgram, "lightfieldIn");
	glEnableVertexAttribArray(lightfieldBufferLocation);
	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldBuffer);
	glVertexAttribPointer(lightfieldBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	// vao - instanced positions
	const auto instancedPositionBufferLocation = 2; // glGetAttribLocation(mGlProgram, "instancedPosition");
	glEnableVertexAttribArray(instancedPositionBufferLocation);
	glBindBuffer(GL_ARRAY_BUFFER, mVertexBuffer);
	glVertexAttribPointer(instancedPositionBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
	glVertexAttribDivisor(instancedPositionBufferLocation, 1);

	// vao - lf complete data
	const auto completeLfBufferLocation = 3;
	glBindBuffer(GL_SHADER_STORAGE_BUFFER, mCompleteLightfieldBuffer);
	glBindBufferBase(GL_SHADER_STORAGE_BUFFER, completeLfBufferLocation, mCompleteLightfieldBuffer);

	glBindVertexArray(0);
}

void SphereRenderer::generateLightfield(unsigned short levels)
{
	// some aliases for readability
	using namespace Generator::Sampler;
	using namespace glm;
	using Plane = PlaneSampler::Plane;

	// create Sampler
	//Plane plane(vec3(0, 0, -10), vec3(0, 0, 1), vec3(0, 1, 0), vec3(1, 0, 0)); // mind the backface culling!
	//const auto planeSampler = make_shared<CheckerboardSampler>(0.f, vec3(1.0f, 0.f, 0.f), plane);
	Plane plane(vec3(0, -1, 0), vec3(0, 1, 0), vec3(0, 0, 1), vec3(1, 0, 0)); // mind the backface culling!
	const auto planeSampler = make_shared<CheckerboardSampler>(5.0f, vec3(0.1f), plane);
	//const auto planeSampler = make_shared<TexturedPlaneSampler>("E:\\crohmann\\tmp\\world_texture.jpg", 2.5f, vec3(0.1f), plane);

	mLightfield = make_unique<Generator::Lightfield>(mSphereData, levels, planeSampler);
}
