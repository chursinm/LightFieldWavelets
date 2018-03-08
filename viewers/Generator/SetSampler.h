#pragma once
#include "Sampler.h"
#include <vector>

namespace Generator
{
	namespace Sampler
	{
		class SetSampler :
			public Sampler
		{
		public:
			SetSampler(const std::vector<std::shared_ptr<Sampler>>& samplerSet, glm::vec3 missColor = glm::vec3(0.f));
			~SetSampler() = default;
			virtual glm::vec4 sample(const Ray& ray) const override;
			std::vector<std::shared_ptr<Sampler>> mSamplers;
			glm::vec3 mMissColor;
		};
	}
}