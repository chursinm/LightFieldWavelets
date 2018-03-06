#include "stdafx.h"
#include "CameraArrayParser.h"
#include "thirdparty\tinyxml2\tinyxml2.h"
#include <experimental\filesystem>

CameraArrayParser::CameraArrayParser()
{
}

CameraArrayParser::~CameraArrayParser()
{
}

CameraArray CameraArrayParser::parse(const std::string & filename)
{
	//TODO some error handling while parsing
	auto path = std::experimental::filesystem::path(filename);
	if(path.empty())
	{
		
	}

	tinyxml2::XMLDocument document;
	document.LoadFile(filename.c_str());
	
	std::vector<std::unique_ptr<ArrayCamera>> cameras;
	glm::vec2 minUv(FLT_MAX, FLT_MAX), maxUv(-FLT_MAX, -FLT_MAX);

	const auto lightfieldElementKey = "lightfield";
	const auto cameraElementKey = "subaperture";
	const auto texturePathAttributeKey = "src";
	const auto uAttributeKey = "u";
	const auto vAttributeKey = "v";
	auto lightfieldElement = document.FirstChildElement(lightfieldElementKey);
	auto cameraElement = lightfieldElement->FirstChildElement(cameraElementKey);
	while(cameraElement)
	{
		auto texturePath = cameraElement->Attribute(texturePathAttributeKey);
		auto u = static_cast<float>(cameraElement->DoubleAttribute(uAttributeKey));
		auto v = static_cast<float>(cameraElement->DoubleAttribute(vAttributeKey));
		auto camera = std::make_unique<ArrayCamera>();
		camera->tex = std::make_unique<Texture>(path.parent_path().append(texturePath).string());
		camera->uv = glm::vec2(u, v);

		if(u < minUv.x) minUv.x = u;
		if(v < minUv.y) minUv.y = v;
		if(u > maxUv.x) maxUv.x = u;
		if(v > maxUv.y) maxUv.y = v;

		cameras.push_back(std::move(camera));
		cameraElement = cameraElement->NextSiblingElement(cameraElementKey);
	}

	// assuming squared grid
	auto gridWidthUnchecked = std::sqrt(cameras.size());
	auto gridWidthUncheckedInt = static_cast<unsigned int>(gridWidthUnchecked + 0.5);
	if(gridWidthUncheckedInt * gridWidthUncheckedInt != cameras.size()) { throw std::runtime_error("Camera grid is not a square"); }
	auto gridWidth = gridWidthUncheckedInt;
	auto gridSize = glm::uvec2(gridWidth, gridWidth);

	return { std::move(cameras), gridSize, minUv, maxUv };
}
