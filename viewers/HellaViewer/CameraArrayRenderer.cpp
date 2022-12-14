#include "stdafx.h"
#include "CameraArrayRenderer.h"
#include <glm/gtx/transform.hpp>
#include "ShaderManager.h"
#include "GLUtility.h"

using namespace glm;

//#define DEMO_FILE "../rectified/chess.xml"
#define DEMO_FILE "E:/crohmann/old_input/stanford_chess_lightfield/rectified/chess.xml"
// think in meters
#define AVERAGE_CAMERA_DISTANCE_X 0.2f

CameraArrayRenderer::CameraArrayRenderer(): m_FocalPlane(3.f)
{
	m_CameraArray = CameraArrayParser::parse(DEMO_FILE);
}


CameraArrayRenderer::~CameraArrayRenderer()
{
}

void CameraArrayRenderer::initialize()
{
	// Create GL programs
	auto& psm = ShaderManager::instance();
	m_GlProgram = psm.from("shader/cameraArray.vert", "shader/cameraArray.geom", "shader/cameraArray.frag");
	if(m_GlProgram == 0)
		throw "init failed";

	auto arrayWidthInput = m_CameraArray.maxUV.x - m_CameraArray.minUV.x;
	auto arrayWidthWorld = AVERAGE_CAMERA_DISTANCE_X * static_cast<float>(m_CameraArray.cameraGridDimension.x);
	auto scaleFactor = arrayWidthWorld / arrayWidthInput;

	//									World Transformation			Model Scaling				Set Model Origin to Zero (it can start with absurd transformations)
	m_CameraArrayQuadsModelMatrix = translate(vec3(-1.f,0.f,-1.f)) * scale(vec3(scaleFactor)) * translate(vec3(-m_CameraArray.minUV, 0.f));

	//////////////////////// TODO function create / upload vertex data ////////////////////////////////////////////////////////
	std::vector<glm::vec3> vertices;
	for(auto& cam : m_CameraArray.cameras)
	{
		vertices.push_back(glm::vec3(cam->uv, 0.f));
	}
	auto indiceCount = (m_CameraArray.cameraGridDimension.x - 1u) * (m_CameraArray.cameraGridDimension.y - 1u) * 6u;
	auto gridWidth = m_CameraArray.cameraGridDimension.x;
	std::vector<unsigned short> indices;
	for(auto y = 0u; y < (m_CameraArray.cameraGridDimension.y - 1u); ++y)
	{
		for(auto x = 0u; x < (m_CameraArray.cameraGridDimension.x - 1u); ++x)
		{
			auto topLeft = x + y * gridWidth;
			auto topRight = x + 1u + y * gridWidth;
			auto botLeft = x + (y + 1u) * gridWidth;
			auto botRight = x + 1u + (y + 1u) * gridWidth;
			indices.push_back(topLeft);
			indices.push_back(botLeft);
			indices.push_back(topRight);

			indices.push_back(botLeft);
			indices.push_back(botRight);
			indices.push_back(topRight);
		}
	}

	auto quadCount = (m_CameraArray.cameraGridDimension.x - 1u) * (m_CameraArray.cameraGridDimension.y - 1u);// *4u;
	for(auto i = 0u; i < quadCount; ++i)
	{
		auto offset = (i * 6u * sizeof(GLshort));
		m_IndexOffsets.push_back(offset);
		m_IndexCounts.push_back(6);
	}

	m_VertexBuffer = GLUtility::generateBuffer(GL_ARRAY_BUFFER, vertices, GL_STATIC_DRAW);
	m_IndexBuffer = GLUtility::generateBuffer(GL_ELEMENT_ARRAY_BUFFER, indices, GL_STATIC_DRAW);

	glGenVertexArrays(1, &m_VertexArrayObject);
	glBindVertexArray(m_VertexArrayObject);
	{
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBuffer);
		glBindBuffer(GL_ARRAY_BUFFER, m_VertexBuffer);
			auto vertexBufferLocation = glGetAttribLocation(m_GlProgram, "worldspaceCameraplaneVertex");
			glVertexAttribPointer(vertexBufferLocation, 3, GL_FLOAT, GL_FALSE, 0, NULL);
			glEnableVertexAttribArray(vertexBufferLocation);
	}
	glBindVertexArray(0);
	////////////////////////////////////////////////////////////////////////////////

	for(auto& cam : m_CameraArray.cameras)
	{
		// For now just transfer them all blocking
		cam->tex->asyncTransferToGPU(std::chrono::duration<int>::max());
	}
}

void CameraArrayRenderer::render(const RenderData& renderData)
{
	if(m_FocalPlane < 0.f) m_FocalPlane = 0.001f;
	auto mvp = renderData.viewProjectionMatrix * m_CameraArrayQuadsModelMatrix;
	auto imvp = inverse(renderData.viewProjectionMatrix);

	/// CALCULATE FOCAL PLANE
	auto a4 = m_CameraArrayQuadsModelMatrix * vec4(m_CameraArray.cameras[0]->uv, 0.f, 1.f);
	auto cR = vec3(0.f, 0.f, -1.f);
	auto a = vec3(a4.xyz * (1.f / a4.w));
	auto worldspaceFocalPlanePosition = a + cR * m_FocalPlane;
	auto worldspaceFocalPlaneDirection = -cR;
	/// END CALCULATE FOCAL PLANE



	glUseProgram(m_GlProgram);
	glDisable(GL_CULL_FACE);

	glUniform3fv(glGetUniformLocation(m_GlProgram, "worldspaceEyePosition"), 1, &renderData.eyePositionWorld[0]);
	glUniformMatrix4fv(glGetUniformLocation(m_GlProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);
	glUniformMatrix4fv(glGetUniformLocation(m_GlProgram, "ivp"), 1, GL_FALSE, &imvp[0][0]);
	glUniform2uiv(glGetUniformLocation(m_GlProgram, "cameraGridDimension"), 1, &m_CameraArray.cameraGridDimension[0]);
	glUniform3fv(glGetUniformLocation(m_GlProgram, "worldspaceFocalPlanePosition"), 1, &worldspaceFocalPlanePosition[0]);
	glUniform3fv(glGetUniformLocation(m_GlProgram, "worldspaceFocalPlaneDirection"), 1, &worldspaceFocalPlaneDirection[0]);
	glUniform1fv(glGetUniformLocation(m_GlProgram, "focalPlaneDistance"), 1, &m_FocalPlane);

	glBindVertexArray(m_VertexArrayObject); 

	auto gridQuadWidth = (m_CameraArray.cameraGridDimension.x - 1u);
	auto quadCount = gridQuadWidth * (m_CameraArray.cameraGridDimension.y - 1u);// *4u;
	for(auto i = 0u; i < quadCount; ++i)
	{
		auto offset = m_IndexOffsets[i];
		auto cameraID = (1u + i) + (i / gridQuadWidth);
		m_CameraArray.cameras[cameraID]->tex->bind(0u);
		glUniform1ui(glGetUniformLocation(m_GlProgram, "quadID"), i);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_SHORT, (void*)offset);
	}

	//glMultiDrawElements(GL_QUADS, &m_IndexCounts[0], GL_UNSIGNED_SHORT, (void**)&m_IndexOffsets[0], (m_CameraArray.cameraGridDimension.x - 1u) * (m_CameraArray.cameraGridDimension.y - 1u));

	glBindVertexArray(0);

	glEnable(GL_CULL_FACE);
	glUseProgram(0);
}

void CameraArrayRenderer::handleInput(SDL_Keymod keymod, SDL_Keycode keycode)
{
	auto cameraSpeedInSceneUnitPerMS = 0.01f;
	if(keymod & (SDL_Keymod::KMOD_LSHIFT | SDL_Keymod::KMOD_RSHIFT)) cameraSpeedInSceneUnitPerMS *= 10.f;
	switch(keycode)
	{
	case SDLK_o:
		m_FocalPlane += cameraSpeedInSceneUnitPerMS;
		std::cout << "Focal plane: " << m_FocalPlane << std::endl;
		break;
	case SDLK_l:
		m_FocalPlane -= cameraSpeedInSceneUnitPerMS;
		std::cout << "Focal plane: " << m_FocalPlane << std::endl;
		break;
	}
}


#undef DEMO_FILE