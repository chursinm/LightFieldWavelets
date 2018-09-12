#pragma once
#include "SphereVertex.h"

class SphereVertex {};

namespace LightField
{
	class SphereFace
	{
	public:
		int					index			= 0;
		SphereVertex*		vertices[3]		= { nullptr, nullptr, nullptr };
		SphereFace*			childFaces[4]	= { nullptr, nullptr, nullptr, nullptr };
	};
	

}
