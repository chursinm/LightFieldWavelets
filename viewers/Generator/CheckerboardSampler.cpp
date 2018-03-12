#include "stdafx.h"
#include "CheckerboardSampler.h"

using namespace Generator::Sampler;

CheckerboardSampler::CheckerboardSampler(float uvScale,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mUvScale(uvScale), mMissColor(missColor)
{
}

glm::vec4 CheckerboardSampler::sample(const Ray& ray) const
{
	using namespace glm;
	float distance = 0.f;
	auto color = glm::vec4(mMissColor, -1.f);
	if(intersect(ray, distance))
	{
		glm::mat3x3 tangentMatrix(mPlane.mTangent, mPlane.mBiTangent, mPlane.mNormal);
		auto iTangentMatrix = glm::inverse(tangentMatrix);

		auto worldIntersection = ray.atDistance(distance);
		//auto c = distance / 20.f;
		//color = glm::vec4(c, c, c, distance);

		float x = worldIntersection.x;
		float y = worldIntersection.z;

		vec3 checkercolor(mod(x, 10.f) >= 5 != mod(y, 10.f) >= 5);
		return vec4(checkercolor, distance);
	}
	return color;
}
