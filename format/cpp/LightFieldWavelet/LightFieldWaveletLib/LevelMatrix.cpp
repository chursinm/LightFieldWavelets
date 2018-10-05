#include "stdafx.h"
#include "LevelMatrix.h"

namespace LightField
{
	LevelMatrix::LevelMatrix(int dim) :
		size(dim)
	{
		data.resize(dim*dim);
	}
	void LevelMatrix::setValue(glm::vec3 vec, int i, int j)
	{
		data[i*size + j] = vec;
	}
	glm::vec3 LevelMatrix::getValue(const int i, const int j)
	{
		return data[i*size + j];
	}
}
