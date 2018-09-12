#include "stdafx.h"
#include "SphereLevel.h"
#include <math.h>

namespace LightField
{
	SphereLevel::SphereLevel(SphereLevel* prevLevel):index(prevLevel->index+1)
	{
		vertices.resize	(10 *	(int)pow(4, index)		+ 2);
		edges.resize	(30 *	(int)pow(4, index)		);
		faces.resize	(20 *	(int)pow(4, index)		);
	}
	
	SphereLevel::SphereLevel(const SphereLevel & sl):index(sl.index), 
		vertices(sl.vertices), 
		edges(sl.edges),
		faces(sl.faces)
	{
		
	}
}