#pragma once
//TODO	Use this as base class for our renderers.
//		Need to decouple RenderContexts user input handling beforehand.. :(
class Renderer
{
public:
	virtual ~Renderer() = 0;
	virtual bool initialize() = 0;
	//void update();
	virtual void render(glm::mat4x4 viewProjection, glm::vec3 eyePosition) = 0;
};

