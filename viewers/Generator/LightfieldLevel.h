#pragma once
#include "SubdivisionSphere.h"
#include "Sampler.h"

namespace Generator
{
	class LightfieldLevel
	{
	public:
		explicit LightfieldLevel(std::shared_ptr<LightField::SubdivisionSphere> sphere, unsigned level, const Sampler::Sampler& sampler);
		~LightfieldLevel() = default;
		std::shared_ptr<std::vector<glm::vec3>> rawData() const;
		unsigned short positionSphereLevel() const;
		unsigned short rotationSphereLevel() const;
		// returns the camera direction for every position
		std::vector<glm::vec3> snapshot(const glm::vec3& cameraPositionInPositionSphereSpace) const;
	private:
		std::shared_ptr<LightField::SubdivisionSphere> mSphere;
		// Layout (position p, rotation r, level-size w): index = p * w + r
		std::shared_ptr<std::vector<glm::vec3>> mRawData;
		unsigned short mPositionSphereLevel;
		unsigned short mRotationSphereLevel;

	};

}
