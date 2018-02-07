#pragma once
#include "Texture.h"
struct ArrayCamera
{
	glm::vec2 uv;
	std::unique_ptr<Texture> tex;
};
struct CameraArray
{
	std::vector<std::unique_ptr<ArrayCamera>> cameras;
	//std::unique_ptr<Texture> depthTex; // not included atm
	glm::uvec2 cameraGridDimension;
	glm::vec2 minUV, maxUV;
};
class CameraArrayParser
{
public:
	static CameraArray parse(const std::string& filename);
private:
	CameraArrayParser();
	~CameraArrayParser();
};

