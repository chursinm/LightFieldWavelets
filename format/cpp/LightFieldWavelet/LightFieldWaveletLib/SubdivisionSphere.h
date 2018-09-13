#pragma once
#include "SphereLevel.h"


namespace LightField
{

	class SubdivisionSphere {
	public:
		SubdivisionSphere(int numberOfLevels);
		size_t getNumberOfLevels() { return levels.size(); }
		const SphereLevel& getLevel(int i) { return levels[i];}
		const std::vector<SphereLevel>& getLevels() { return levels; } 
		int vectorToFaceIndex(const glm::vec3& vector, int level);
		glm::vec3 indexToVector(const int index, int level) { return levels[level].vertices[index].pos; }

	private:
		std::vector<SphereLevel> levels;

		static bool sameSide(const glm::vec3 A1, const glm::vec3 A2, const glm::vec3 testPoint, const glm::vec3 point);
		static bool testTriangle(const glm::vec3 A, const glm::vec3 B, const glm::vec3 C, const glm::vec3 point);


	};
}