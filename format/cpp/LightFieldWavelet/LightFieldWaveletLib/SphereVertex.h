#pragma once


#include <glm/glm.hpp>
#include "SphereEdge.h"


namespace LightField {

	class SphereVertex{
	public:
		int				index				= 0;
		glm::vec3		pos					= glm::vec3(0.0f, 0.0f, 0.0f);		


		SphereEdge*		edges[6]			= { nullptr,
												nullptr,
												nullptr,
												nullptr,
												nullptr,
												nullptr };
		SphereFace*		faces[6]			 = { nullptr,
												nullptr,
												nullptr,
												nullptr,
												nullptr,
												nullptr };

		//prev level		
		SphereEdge*		parentEdge = nullptr;
		SphereVertex*	directParentVertex = nullptr; //in case of direct parent

		SphereVertex& operator = (const SphereVertex& other)
		{
			if (this != &other)
			{
				pos = other.pos;
				index = other.index;
				parentEdge = other.parentEdge;
			}
			return *this;
		}


	};
}

