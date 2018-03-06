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
	void render(const RenderData& renderData) override;
	
private:
	GLuint mGlProgram;
};

