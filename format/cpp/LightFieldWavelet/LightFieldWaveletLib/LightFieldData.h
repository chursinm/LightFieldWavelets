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
		std::shared_ptr<LevelMatrix> getLevelMatrix(int level) const { return levelMatices[level]; } 
	private:
		std::shared_ptr <SubdivisionSphere> subSphere;
		std::vector <std::shared_ptr<LevelMatrix>> levelMatices;
		
	};
}