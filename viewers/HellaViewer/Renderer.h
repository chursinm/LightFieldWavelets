#pragma once
//TODO	Use this as base class for our renderers.
//		Need to decouple RenderContexts user input handling beforehand.. :(
class Renderer
{
public:
	/*
		Cleanup used (opengl) resources
	*/
	virtual ~Renderer() {};
	/*
		Called once after opengl is set up
	*/
	virtual void initialize() = 0;
	/*
		Called once per frame
	*/
	virtual void update(double timestep) = 0;
	/*
		Called per viewport
	*/
	virtual void render(const glm::mat4x4& viewProjection, const glm::vec3& eyePosition) = 0;
};
