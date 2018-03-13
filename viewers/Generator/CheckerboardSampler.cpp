#include "stdafx.h"
#include "CheckerboardSampler.h"

using namespace Generator::Sampler;

CheckerboardSampler::CheckerboardSampler(float checkerSquareLength,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mCheckerSquareLength(checkerSquareLength), mMissColor(missColor)
{
}

glm::vec4 CheckerboardSampler::sample(const Ray& ray) const
{
	using namespace glm;
	float distance = 0.f;
	auto color = glm::vec4(mMissColor, -1.f);
	if(intersect(ray, distance))
	{
		const auto worldIntersection = ray.atDistance(distance);
		const auto tangentIntersection = mPlane.mTangentMatrix * worldIntersection;
		//auto c = distance / 20.f;
		//color = glm::vec4(c, c, c, distance);


		const auto doubleSquareLength = 2.0f * mCheckerSquareLength;
		const vec3 checkercolor(mod(tangentIntersection.x, doubleSquareLength) >= mCheckerSquareLength != mod(tangentIntersection.y, doubleSquareLength) >= mCheckerSquareLength);

		return vec4(checkercolor*0.8f, distance);
	}
	return color;
}
