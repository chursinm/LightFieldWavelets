#pragma once
#include "LightfieldLevel.h"
#include <future>

namespace Generator
{
	class Lightfield
	{
	public:
		Lightfield(std::shared_ptr<LightField::SubdivisionSphere> sphere, unsigned levels, std::shared_ptr<Sampler::Sampler> sampler);
		~Lightfield() = default;
		const LightfieldLevel& level(unsigned i);
	private:
		std::vector<std::future<std::unique_ptr<LightfieldLevel>>> mLevelFutures;
		std::vector<std::unique_ptr<LightfieldLevel>> mLevels;
	};
}
