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
	mSphereData(make_shared<LightField::SubdivisionSphere>(levelCount)), mFacesCount(0), mCurrentLevel(0), mRenderMode(RenderMode::POSITION)
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

	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldSliceBuffer);
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
	auto face = sphereLevel.getFaces()[0];
	auto posA = face.vertices[0]->pos;
	auto posB = face.vertices[1]->pos;
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
	const int numberOfVertices = sphereLevel.getNumberOfVertices();
	GLUtility::setUniform(mRotationSphereGlProgram, "vertexCount", numberOfVertices);
	GLUtility::setUniform(mRotationSphereGlProgram, "visualize_lightfield", mRenderMode == RenderMode::LIGHTFIELD);

	glBindVertexArray(mVertexArrayObject);
	glDrawElementsInstanced(GL_TRIANGLES, mFacesCount * 3u, GL_UNSIGNED_INT, reinterpret_cast<void*>(0), sphereLevel.getNumberOfVertices());
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
		if(i < sphereLevel.getNumberOfVertices())
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
		if(i < sphereLevel.getNumberOfFaces())
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
	switch(mRenderMode)
	{
	case RenderMode::POSITION:
		renderPositionSphere(renderData);
		break;
	case RenderMode::ROTATION:
	case RenderMode::LIGHTFIELD:
		renderRotationSpheres(renderData);
		break;
	case RenderMode::LIGHTFIELD_SLICE:
		renderLightfield(renderData);
		break;
	default:
		WARN_ONCE("no valid rendermode set")
		break;
	}
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
	glDeleteBuffers(1, &mLightfieldSliceBuffer);
	glDeleteBuffers(1, &mCompleteLightfieldBuffer);
	glDeleteVertexArrays(1, &mVertexArrayObject);
}

void SphereRenderer::setupGlBuffersForLevel(unsigned short level) 
{
	auto sphereLevel = mSphereData->getLevel(level);
	mFacesCount = sphereLevel.getNumberOfFaces();

	// Upload sphere indices
	{
		auto indices = make_unique<vector<unsigned int>>(); // shorts are not enough :o 
		indices->reserve(sphereLevel.getNumberOfFaces() * 3);
		for(auto faceIterator : sphereLevel.getFaces())
		{
			indices->push_back(faceIterator.vertices[0]->index);
			indices->push_back(faceIterator.vertices[1]->index);
			indices->push_back(faceIterator.vertices[2]->index);
		}
		mIndexBuffer = GLUtility::generateBuffer(GL_ELEMENT_ARRAY_BUFFER, *indices, GL_STATIC_DRAW);
		// TODO direct transfer using bufferSubData
	}

	// Upload sphere vertices
	{
		auto vertices = make_unique<vector<glm::vec3>>();
		vertices->reserve(sphereLevel.getNumberOfVertices());
		for(auto vertexIterator : sphereLevel.getVertices())
		{
			vertices->push_back(vertexIterator.pos);
		}
		mVertexBuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, *vertices, GL_STATIC_DRAW);
		// TODO direct transfer using bufferSubData
	}

	// Allocate lightfield memory
	{
		mLightfieldSliceBuffer = GLUtility::generateBuffer<glm::vec3>(GL_ARRAY_BUFFER, sphereLevel.getNumberOfVertices(), nullptr, GL_DYNAMIC_DRAW);

		const auto lfData = mLightfield->level(mCurrentLevel).rawData();
		std::vector<glm::vec4> vec4Lightfield;
		vec4Lightfield.reserve(lfData->size());
		for(auto i : *lfData) vec4Lightfield.push_back(glm::vec4(i.rgb, 0.0f));
		mCompleteLightfieldBuffer = GLUtility::generateBuffer(GL_SHADER_STORAGE_BUFFER, vec4Lightfield, GL_DYNAMIC_DRAW);
		glMemoryBarrier(GL_VERTEX_ATTRIB_ARRAY_BARRIER_BIT); // TODO: this is a precaution, not 100% sure about this barrier. need more intel on SSBs
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
	glBindBuffer(GL_ARRAY_BUFFER, mLightfieldSliceBuffer);
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
