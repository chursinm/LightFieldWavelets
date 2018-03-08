#include "stdafx.h"
#include "ColoredPlaneSampler.h"

ColoredPlaneSampler::ColoredPlaneSampler(const glm::vec3& hitColor, const glm::vec3& missColor, const Plane& plane):
	PlaneSampler(plane), mMissColor(missColor), mHitColor(hitColor)
{
}

glm::vec3 ColoredPlaneSampler::sample(const Ray& ray) const
{
	return this->intersect(ray) ? mHitColor : mMissColor;
}
