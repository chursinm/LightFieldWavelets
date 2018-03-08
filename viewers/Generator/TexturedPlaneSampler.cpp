#include "stdafx.h"
#include "TexturedPlaneSampler.h"

using namespace Generator::Sampler;

TexturedPlaneSampler::TexturedPlaneSampler(const std::string& texturePath, const float uvScale,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mTexture(std::make_unique<SDL_Surface>(/*TODO*/)), mUvScale(0), mMissColor(missColor)
{
	throw std::exception("not yet implemented");
}

glm::vec4 TexturedPlaneSampler::sample(const Ray& ray) const
{
	throw std::exception("not yet implemented");
	float distance = 0.f;
	auto color = glm::vec4(mMissColor, -1.f);
	if(intersect(ray, distance))
	{
		color = glm::vec4(1.f, 0.f, 0.f, distance);
	}
	return color;
}
