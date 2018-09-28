#pragma once
#include "SphereVertex.h"



namespace LightField
{
	class SphereVertex;
	class SphereEdge;
	class SphereFace
	{
	public:
		int					index			= 0;
		SphereVertex*		vertices[3]		= { nullptr, nullptr, nullptr };		
		SphereEdge*			edges[3]		= { nullptr, nullptr, nullptr };

		//next level
		SphereFace*			childFaces[4]	= { nullptr, nullptr, nullptr, nullptr };
		//prev level
		SphereFace*			parentFace		=   nullptr;
	};
	

}
