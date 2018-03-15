#include "stdafx.h"
#include "CheckerboardSampler.h"

using namespace Generator::Sampler;
using namespace glm;

CheckerboardSampler::CheckerboardSampler(float checkerSquareLength,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mCheckerSquareLength(checkerSquareLength), mMissColor(missColor)
{
}

float checkersTextureGradBox(const vec2& p, const vec2& ddx, const vec2& ddy)
{ // credits: http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
  // filter kernel
	vec2 w = max(abs(ddx), abs(ddy)) + 0.01f;
	// analytical integral (box filter)
	vec2 i = 2.0f*(abs(fract((p - 0.5f*w) / 2.0f) - 0.5f) - abs(fract((p + 0.5f*w) / 2.0f) - 0.5f)) / w;
	// xor pattern
	return 0.5f - 0.5f*i.x*i.y;
}

float checkersTexture(const vec2& p)
{ // credits: http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
	vec2 q = floor(p);
	return mod(q.x + q.y, 2.0f);            // xor pattern
}

glm::vec4 CheckerboardSampler::sample(const Generator::Ray& ray) const
{
	using namespace glm;
	float distance = 0.f;
	auto color = glm::vec4(mMissColor, -1.f);
	if(intersect(ray, distance))
	{
		const auto worldIntersection = ray.atDistance(distance);
		const auto tangentIntersection = mPlane.mTangentMatrix * worldIntersection;

		vec3 checkercolor(checkersTexture(tangentIntersection.xy));
		/*float dA, dB;
		if(ray.a && ray.b && intersect(*ray.a, dA) && intersect(*ray.b, dB))
		{
			const auto tia = mPlane.mTangentMatrix * ray.a->atDistance(dA);
			const auto tib = mPlane.mTangentMatrix * ray.b->atDistance(dB);
			const auto& uv = tangentIntersection.xy;
			const auto ddxUv = tia.xy - uv;
			const auto ddyUv = vec2(-ddxUv.x, ddxUv.y);//tib.xy - uv;

			checkercolor = vec3(checkersTextureGradBox(uv, ddxUv, ddyUv));
		}*/

		return vec4(checkercolor, distance);
	}
	return color;
}
