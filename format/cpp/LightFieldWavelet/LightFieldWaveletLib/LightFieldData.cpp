#include "stdafx.h"
#include "LightFieldData.h"

namespace LightField
{
	LightFieldData::LightFieldData(const std::shared_ptr<SubdivisionSphere> subSpheresIn):
		subSphere(subSpheresIn)
	{
		for (auto& level : subSphere->getLevels())
		{
			levelMatices.push_back(std::make_shared<LevelMatrix>(level.getNumberOfVertices()));
		}
	}
}