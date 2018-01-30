#pragma once
#include "CameraArrayParser.h"
class CameraArrayRenderer
{
public:
	CameraArrayRenderer();
	~CameraArrayRenderer();
	bool initialize();
	//void update();
	void render(glm::mat4x4 viewProjection, glm::vec3 eyePosition);
private:
	CameraArray m_CameraArray;
	GLuint m_VertexBuffer, m_IndexBuffer;
	glm::mat4x4 m_CameraArrayQuadsModelMatrix;
	GLuint m_GlProgram;
};

