#include "stdafx.h"
#include "PlaneSampler.h"


bool PlaneSampler::intersect(const Ray& ray, glm::vec3& intersectionPoint) const
{
	auto iDistance = 0.f;
	const auto intersects = glm::intersectRayPlane(ray.mOrigin, ray.mDirection, mPlane.mOrigin, mPlane.mNormal, iDistance) && iDistance >= 0.f;
	if(intersects)
	{
		intersectionPoint = ray.mOrigin + ray.mDirection * iDistance;
	}
	return intersects;
}

bool PlaneSampler::intersect(const Ray& ray) const
{
	glm::vec3 devNull;
	return intersect(ray, devNull);
}
