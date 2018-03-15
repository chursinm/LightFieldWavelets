#include "stdafx.h"
#include "TexturedPlaneSampler.h"

using namespace Generator::Sampler;

TexturedPlaneSampler::TexturedPlaneSampler(const std::string& texturePath, const float uvScale,
	const glm::vec3& missColor, const Plane& plane) : PlaneSampler(plane), mTexture(IMG_Load(texturePath.c_str())), mUvScale(uvScale), mMissColor(missColor)
{
}

glm::vec4 TexturedPlaneSampler::sample(const Generator::Ray& ray) const
{
	using namespace glm;
	float distance = 0.f;
	auto color = glm::vec4(mMissColor, -1.f);
	if(intersect(ray, distance))
	{
		const auto worldIntersection = ray.atDistance(distance);
		const auto tangentIntersection = mPlane.mTangentMatrix * worldIntersection;

		const auto u = mod(tangentIntersection.x, mUvScale) / mUvScale;
		const auto v = mod(tangentIntersection.y, mUvScale) / mUvScale;
		const auto x = static_cast<unsigned>(u * mTexture->w);
		const auto y = static_cast<unsigned>(v * mTexture->h);
		const auto i = x * mTexture->pitch + y;
		// TODO cast pixels pointer to specific type using sdl_pixelformat
		// TODO mipmaping or sat
		//const auto color = mTexture->pixels[i];

		return vec4(vec3(u,v,0.f), distance);
	}
	return color;
}
