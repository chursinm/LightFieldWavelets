#include "stdafx.h"
#include "TexturedPlaneSampler.h"

TexturedPlaneSampler::TexturedPlaneSampler(const std::string& texturePath, const float uvScale,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mTexture(std::make_unique<SDL_Surface>(/*TODO*/)), mUvScale(0), mMissColor(missColor)
{
}

glm::vec3 TexturedPlaneSampler::sample(const Ray& ray) const
{
	glm::vec3 intersectionPoint(0.0f);
	auto color = mMissColor;
	if(intersect(ray, intersectionPoint))
	{
		color = glm::vec3(1.f, 0.f, 0.f);
	}
	return color;
}
