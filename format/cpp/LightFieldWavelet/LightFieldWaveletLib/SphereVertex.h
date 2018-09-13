#pragma once


#include <glm/glm.hpp>
#include "SphereEdge.h"


namespace LightField {

	class SphereVertex{
	public:
		int				index				= 0;
		glm::vec3		pos					= glm::vec3(0.0f, 0.0f, 0.0f);		
		SphereVertex*	parentVertices[2]	= { nullptr, nullptr };
		SphereEdge*		parentEdge			= nullptr;
	};
}

