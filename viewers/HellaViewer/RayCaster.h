#pragma once
#include "Ray.h"
#include "TrackballCamera.h"
class RayCaster
{
public:
	RayCaster();
	~RayCaster();
private:
	glm::uvec2 mResolution;
	TrackballCamera mCamera;
	std::unique_ptr<std::vector<glm::vec3>> mImage;
	Generator::Ray generateRay(const glm::uvec2& imageCoords)
	{
		using namespace glm;
		// maybe save them directly as float? o.O
		const auto x = static_cast<float>(imageCoords.x);
		const auto y = static_cast<float>(imageCoords.y);
		const auto height = static_cast<float>(mResolution.y);
		const auto width = static_cast<float>(mResolution.x);
		const auto inverseView = inverse(mCamera.viewMatrix());

		const glm::vec4 originViewSpace(0.f, 0.f, 0.f, 1.f);
		const glm::vec4 originWorldSpace = inverseView * originViewSpace;

		const float z = height / (std::tan(float(M_PI) / 180.f * mCamera.fovy()));
		const glm::vec4 directionViewSpace(glm::normalize(glm::vec3(
			(x - width / 2.f), y - height / 2.f, -z)), 0.f);
		const glm::vec4 directionWorldSpace = inverseView * directionViewSpace;

		return Generator::Ray(glm::vec3(originWorldSpace), glm::vec3(directionWorldSpace));
	}
};
