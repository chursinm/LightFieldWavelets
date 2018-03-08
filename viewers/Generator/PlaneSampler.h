#pragma once
#include "Sampler.h"
class PlaneSampler :
	public Sampler
{
public:
	struct Plane
	{
		Plane(const glm::vec3& origin, const glm::vec3& normal, const glm::vec3& tangent, const glm::vec3& biTangent)
			: mOrigin(origin),
			mNormal(glm::normalize(normal)),
			mTangent(glm::normalize(tangent)),
			mBiTangent(glm::normalize(biTangent))
		{
		}
		Plane(const glm::vec3& origin, const glm::vec3& normal, const glm::vec3& tangent)
			: Plane(origin, normal, tangent, glm::cross(glm::normalize(normal), glm::normalize(tangent)))
		{
		}
		glm::vec3 mOrigin, mNormal, mTangent, mBiTangent;
	};

protected:
	explicit PlaneSampler(const Plane& plane) : mPlane(plane) {}
	bool intersect(const Ray& ray, glm::vec3& intersectionPoint) const;
	bool intersect(const Ray& ray) const;
	Plane mPlane;
};
using PlaneSamplerSamplerPtr = std::unique_ptr<PlaneSampler>;