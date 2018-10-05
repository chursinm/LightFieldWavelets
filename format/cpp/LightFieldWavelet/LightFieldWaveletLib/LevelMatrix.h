#pragma once


namespace LightField
{
	class LevelMatrix
	{
	public:
		LevelMatrix(int dim);
		void setValue(glm::vec3, int i, int j);
		glm::vec3 getValue(const int i, const int j);
	private:
		std::vector<glm::vec3> data;
		int size;
	};
}
