#include "stdafx.h"
#include "LevelMatrix.h"
#include <iostream>

namespace LightField
{
	LevelMatrix::LevelMatrix(int dim) :
		size(dim)
	{
		size_t dim_s = dim;

		std::cout <<"size of light field matrix in Gb " << dim_s * dim_s * sizeof(glm::vec3)/(1024*1024*1024)<< std::endl;

		data.resize(dim_s*dim_s);

	}
	void LevelMatrix::setValue(glm::vec3 vec, int  i, int  j)
	{
		size_t dim_i = i;
		size_t dim_size = size;
		size_t dim_j = j;
		data[dim_i *dim_size+ dim_j] = vec;
	}
	void LevelMatrix::addValue(glm::vec3 vec, int i, int j)
	{
		size_t dim_i = i;
		size_t dim_size = size;
		size_t dim_j = j;
		data[dim_i *dim_size + dim_j] += vec;

	}
	glm::vec3 LevelMatrix::getValue(const int i, const int j)
	{
		return data[i*size + j];
	}
}
