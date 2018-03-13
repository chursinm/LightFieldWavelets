#pragma once
#include "SubdivisionSphere.h"
#include "Sampler.h"

namespace Generator
{
	class LightfieldLevel
	{
	public:
		explicit LightfieldLevel(std::shared_ptr<SubdivisionShpere::SubdivisionSphere> sphere, unsigned int level, const Sampler::Sampler& sampler);
		~LightfieldLevel() = default;
		std::shared_ptr<std::vector<glm::vec3>> rawData();
		// returns the camera direction for every position
		std::vector<glm::vec3> snapshot(const glm::vec3& cameraPositionInPositionSphereSpace) const;
	private:
		std::shared_ptr<SubdivisionShpere::SubdivisionSphere> mSphere;
		unsigned int mLevel;
		// Layout (position p, rotation r, level-size w): index = p * w + r
		std::shared_ptr<std::vector<glm::vec3>> mRawData;

	};

}
