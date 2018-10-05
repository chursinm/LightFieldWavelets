#pragma once
#include "SphereFace.h"
#include "SphereVertex.h"

namespace LightField {
	class SphereVertex;
	class SphereFace;
	class SphereEdge {
	public:
		int				index			= 0;
		SphereVertex*	vertices[2]		= { nullptr, nullptr };
		SphereFace*		faces[2]		= { nullptr, nullptr };

		//next level
		SphereVertex*	childVertex		=	nullptr;
		
		
	};

}
