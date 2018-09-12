#pragma once
#include <vector>
#include "SphereVertex.h"

namespace LightField
{
	class SphereLevel
	{
	public:
		SphereLevel(SphereLevel* prevLever);		
		SphereLevel(const SphereLevel & sl);
		SphereLevel():index(0){};

	protected:
		std::vector <SphereVertex>	vertices;
		std::vector <SphereEdge>	edges;
		std::vector <SphereFace>	faces;
		int index = 0;
	};
}