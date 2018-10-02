#include "stdafx.h"
#include "Lightfield.h"


Generator::Lightfield::Lightfield(const std::shared_ptr<LightField::SubdivisionSphere> sphere, const unsigned levels, const std::shared_ptr<Sampler::Sampler> sampler): mLevels(levels)
{
	for(auto i = 0u; i < levels; ++i)
	{
		//mLevels.push_back(std::make_unique<LightfieldLevel>(sphere, i, sampler));
		const auto launchtype = i > 6 ? std::launch::deferred : std::launch::async;
		mLevelFutures.push_back(std::async(launchtype, [sphere,sampler,i]()
		{
			return std::make_unique<LightfieldLevel>(sphere, i, *sampler);
		}));
		
	}
}

const Generator::LightfieldLevel& Generator::Lightfield::level(const unsigned i)
{
	if(!mLevels[i])
	{
		mLevels[i] = mLevelFutures[i].get();
	}
	return *mLevels[i];
}
