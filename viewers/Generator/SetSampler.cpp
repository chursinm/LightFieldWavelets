#include "stdafx.h"
#include "SetSampler.h"

using namespace Generator::Sampler;

SetSampler::SetSampler(const std::vector<std::shared_ptr<Sampler>>& samplerSet, glm::vec3 missColor): mSamplers(samplerSet), mMissColor(missColor)
{
}

glm::vec4 SetSampler::sample(const Ray& ray) const
{
	glm::vec4 result(mMissColor, 1000.f);
	for(const auto& sampler : mSamplers)
	{
		const auto intermediate = sampler->sample(ray);
		if(intermediate.w >= 0.f && intermediate.w < result.w)
		{
			result = intermediate;
		}
	}
	return result;
}
