#include "stdafx.h"
#include "CameraArrayRenderer.h"
#include <glm/gtx/transform.hpp>
#include "ShaderManager.h"

using namespace glm;

#define DEMO_FILE "E:/crohmann/old_input/stanford_chess_lightfield/preview/chess.xml"
// think in meters
#define AVERAGE_CAMERA_DISTANCE_X 0.4f

CameraArrayRenderer::CameraArrayRenderer()
{
	m_CameraArray = CameraArrayParser::parse(DEMO_FILE);
}


CameraArrayRenderer::~CameraArrayRenderer()
{
}

bool CameraArrayRenderer::initialize()
{
	// Create GL programs
	auto& psm = ShaderManager::instance();
	m_GlProgram = psm.from("shader/cameraArray.vert", "shader/cameraArray.frag");
	if(m_GlProgram == 0)
		return false;

	auto arrayWidthInput = m_CameraArray.maxUV.x - m_CameraArray.minUV.x;
	auto arrayWidthWorld = AVERAGE_CAMERA_DISTANCE_X * static_cast<float>(m_CameraArray.cameraGridDimension.x);
	auto scaleFactor = arrayWidthWorld / arrayWidthInput;

	//m_CameraArrayQuadsModelMatrix = translate(vec3(m_CameraArray.minUV, 0.f)) * scale(vec3(scaleFactor));
	m_CameraArrayQuadsModelMatrix = translate(vec3(0.f,0.f,-10.f)) * scale(vec3(scaleFactor)) * translate(vec3(-m_CameraArray.minUV, 0.f));

	////////////////////////TODO fnction create / upload vertex data ////////////////////////////////////////////////////////
	std::vector<glm::vec3> vertices;
	for(auto& cam : m_CameraArray.cameras)
	{
		vertices.push_back(glm::vec3(cam->uv, 0.f));
	}
	auto indiceCount = (m_CameraArray.cameraGridDimension.x - 1u) * (m_CameraArray.cameraGridDimension.y - 1u) * 4u;
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
			indices.push_back(botRight);
			indices.push_back(topRight);
		}
	}


	glGenBuffers(1, &m_VertexBuffer);
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBuffer);
	{
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(glm::vec3), &vertices[0], GL_STATIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenBuffers(1, &m_IndexBuffer);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBuffer);
	{
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned short), &indices[0], GL_STATIC_DRAW);
	}
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	////////////////////////////////////////////////////////////////////////////////

	for(auto& cam : m_CameraArray.cameras)
	{
		// For now just transfer them all blocking
		cam->tex->asyncTransferToGPU(std::chrono::duration<int>::max());
	}

	return true;
}

void CameraArrayRenderer::render(glm::mat4x4 viewProjection, glm::vec3 eyePosition)
{
	glUseProgram(m_GlProgram);
	glDisable(GL_CULL_FACE);

	//glUniformMatrix4fv(glGetUniformLocation(m_GlProgram, "mvp"), 1, GL_FALSE, &viewProjection[0][0]);
	//glUniformMatrix4fv(glGetUniformLocation(m_GlProgram, "inverse_mvp"), 1, GL_FALSE, &inverse(viewProjection)[0][0]);
	//glUniform3fv(glGetUniformLocation(m_GlProgram, "worldSpace_eyePosition"), 1, &eyePosition[0]);

	auto mvp = viewProjection * m_CameraArrayQuadsModelMatrix;
	glUniformMatrix4fv(glGetUniformLocation(m_GlProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);
	
	auto it = m_CameraArray.cameras.begin();
	auto& firstCamera = *it;
	auto firstCameraUv = firstCamera->uv;
	auto secondCameraUv = (*(it+1))->uv;
	glBindTexture(GL_TEXTURE_2D, firstCamera->tex->textureID());

	glBegin(GL_QUADS);
	glTexCoord2f(0.f,0.f);
	glVertex3f(firstCameraUv.x, firstCameraUv.y, 0.f);
	glTexCoord2f(1.f, 0.f);
	glVertex3f(secondCameraUv.x, firstCameraUv.y, 0.f);
	glTexCoord2f(1.f, 1.f);
	glVertex3f(secondCameraUv.x, secondCameraUv.y, 0.f);
	glTexCoord2f(0.f, 1.f);
	glVertex3f(firstCameraUv.x, secondCameraUv.y, 0.f);
	glEnd();

	glEnableClientState(GL_VERTEX_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, m_VertexBuffer);
	glVertexPointer(3, GL_FLOAT, sizeof(glm::vec3), (void*)0);
	glTexCoordPointer(3, GL_FLOAT, sizeof(glm::vec3), (void*)(sizeof(glm::vec3)));
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_IndexBuffer);

	auto quadCount = (m_CameraArray.cameraGridDimension.x - 1u) * (m_CameraArray.cameraGridDimension.y - 1u);// *4u;
	for(auto i = 0u; i < quadCount; ++i)
		glDrawElements(GL_QUADS, 4, GL_UNSIGNED_SHORT, (void*)(i*4*sizeof(GLshort)));

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);

	glEnable(GL_CULL_FACE);
	glUseProgram(0);
}


#undef DEMO_FILE