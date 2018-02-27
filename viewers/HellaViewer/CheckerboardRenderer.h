#pragma once
#include "Renderer.h"
class CheckerboardRenderer :
	public Renderer
{
public:
	CheckerboardRenderer();
	~CheckerboardRenderer();

	void initialize() override;
	void update(double timestep) override;
	void render(const glm::mat4x4 & viewProjection, const glm::vec3 & eyePosition) override;
	
private:
	GLuint mGlProgram;
};

