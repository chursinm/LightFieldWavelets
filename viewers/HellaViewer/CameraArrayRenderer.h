#pragma once
#include "CameraArrayParser.h"

class CameraArrayRenderer
{
private:
	struct Vertex
	{
		glm::vec3 position;
		glm::vec2 uv;
	};
public:
	CameraArrayRenderer();
	~CameraArrayRenderer();
	bool initialize();
	//void update();
	void render(glm::mat4x4 viewProjection, glm::vec3 eyePosition);
	float m_FocalPlane;
private:
	CameraArray m_CameraArray;
	GLuint m_VertexArrayObject, m_VertexBuffer, m_IndexBuffer;
	std::vector<unsigned long long> m_IndexOffsets;
	std::vector<GLsizei> m_IndexCounts;
	glm::mat4x4 m_CameraArrayQuadsModelMatrix;
	GLuint m_GlProgram, m_GLTextureID;
};

