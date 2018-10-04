#pragma once


namespace LightField
{
	class LevelMatrix
	{
	public:
		LevelMatrix(int dim);
	private:
		std::vector<glm::vec3> data;
	};
}
