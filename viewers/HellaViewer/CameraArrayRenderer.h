#pragma once
#include "Renderer.h"
#include "CameraArrayParser.h"

class CameraArrayRenderer : public Renderer
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
	void initialize() override;
	void render(const RenderData& renderData) override;
	void update(double timestep) override {};
	void handleInput(SDL_Keymod mod, SDL_Keycode code);
private:
	float m_FocalPlane;
	CameraArray m_CameraArray;
	GLuint m_VertexArrayObject, m_VertexBuffer, m_IndexBuffer;
	std::vector<unsigned long long> m_IndexOffsets;
	std::vector<GLsizei> m_IndexCounts;
	glm::mat4x4 m_CameraArrayQuadsModelMatrix;
	GLuint m_GlProgram, m_GLTextureID;
};
