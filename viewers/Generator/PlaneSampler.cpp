#include "stdafx.h"
#include "PlaneSampler.h"

using namespace Generator::Sampler;

bool PlaneSampler::intersect(const Ray& ray, float& intersectionPoint) const
{
	return glm::intersectRayPlane(ray.mOrigin, ray.mDirection, mPlane.mOrigin, mPlane.mNormal, intersectionPoint) && intersectionPoint >= 0.f;
}