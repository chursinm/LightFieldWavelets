#include "stdafx.h"
#include "ColoredPlaneSampler.h"

using namespace Generator::Sampler;

ColoredPlaneSampler::ColoredPlaneSampler(const glm::vec3& hitColor, const glm::vec3& missColor, const Plane& plane):
	PlaneSampler(plane), mMissColor(missColor), mHitColor(hitColor)
{
}

glm::vec4 ColoredPlaneSampler::sample(const Generator::Ray& ray) const
{
	float distance = 0.f;
	if(this->intersect(ray, distance))
	{
		return glm::vec4(mHitColor, distance);
	}
	return glm::vec4(mMissColor, -1.f);
}
