#pragma once
#include "SphereVertex.h"



namespace LightField
{
	class SphereVertex;
	class SphereFace
	{
	public:
		int					index			= 0;
		SphereVertex*		vertices[3]		= { nullptr, nullptr, nullptr };
		SphereFace*			childFaces[4]	= { nullptr, nullptr, nullptr, nullptr };
	};
	

}
