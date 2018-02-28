#pragma once
class Renderer
{
public:

	struct RenderData
	{
		glm::mat4x4 viewProjectionMatrix;
		glm::mat4x4 viewMatrix;
		glm::mat4x4 projectionMatrix;
		glm::vec3 eyePositionWorld;
	};

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
	virtual void render(const RenderData& renderData) = 0;
};
