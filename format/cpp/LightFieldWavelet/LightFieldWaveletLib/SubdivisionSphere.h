#pragma once
#include "SphereLevel.h"
#include "SphereInitialLevel.h"

namespace LightField
{

	class SubdivisionSphere {
	public:
		SubdivisionSphere(int numberOfLevels);
		size_t getNumberOfLevels() { return levels.size(); }
		SphereLevel& getLevel(int i) { return levels.at(i);}
	private:
		std::vector<SphereLevel> levels;

	};
}