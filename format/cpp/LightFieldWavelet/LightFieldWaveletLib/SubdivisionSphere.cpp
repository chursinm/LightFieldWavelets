#include "stdafx.h"
#include "SubdivisionSphere.h"


namespace LightField
{
	SubdivisionSphere::SubdivisionSphere(int numberOfLevels)
	{
		SphereInitialLevel sl;
		levels.push_back(sl);
		for (int i = 1; i < numberOfLevels; i++)
		{			
			levels.push_back(SphereLevel(&levels.back()));
		}
	}
}