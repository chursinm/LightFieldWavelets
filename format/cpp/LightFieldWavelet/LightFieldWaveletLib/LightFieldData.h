#pragma once
#include "SubdivisionSphere.h"
#include "LevelMatrix.h"
#include <memory>



namespace LightField
{
	
	class LightFieldData
	{
	public:
		LightFieldData(const std::shared_ptr <SubdivisionSphere> subsphere);
		
	private:
		std::shared_ptr <SubdivisionSphere> subSphere;
		std::vector <LevelMatrix*> levelMatices;
		
	};
}