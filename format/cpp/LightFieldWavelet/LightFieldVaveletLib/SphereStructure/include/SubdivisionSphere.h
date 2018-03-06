//#pragma once


#include "Constants.h"
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>
#include <glm\vec3.hpp>
#include<glm\glm.hpp>



// Subdivision sphere structure

namespace SubdivisionShpere {

	struct Vertex
	{		
		int index = 0;
		glm::vec3 position = glm::vec3(0.0f,0.0f,0.0f);
		int creationLevel = 0;
		int parentA = 0;
		int parentB = 0;
		Vertex* parentARef = NULL;
		Vertex* parentBRef = NULL;
	};

	struct Face
	{
		int index = 0;
		int vertA = 0;
		int vertB = 0;
		int vertC = 0;

		Vertex* vertARef = 0;
		Vertex* vertBRef = 0;
		Vertex* vertCRef = 0;

		int childFaceA = 0;
		int childFaceB = 0;
		int childFaceC = 0;
		int childFaceD = 0;
		int parentFace = 0;
		Face* childFaceARef = 0;
		Face* childFaceBRef = 0;
		Face* childFaceCRef = 0;
		Face* childFaceDRef = 0;
		Face* parentFaceRef = 0;

	};

	struct Level
	{
		Vertex* vertices = nullptr;
		Face* faces = nullptr;
		int numberOfVertices = 0;
		int numberOfFaces = 0;
		int levelIndex = 0;
	};


	class SubdivisionSphere
	{
	public:
		
		SubdivisionSphere();
		SubdivisionSphere(int numberOfLevels);
		~SubdivisionSphere();
		int getNumberOfLevels() { return _numberOfLevels; }
		Level getLevel(int levelIndex) { return levels[levelIndex]; }
		int vectorToFaceIndex(const glm::vec3& vector, int level);
		glm::vec3 indexToVector(const int index, int level) { return levels[level].vertices[index].position; }

		std::vector<std::vector<Face>> findAllAjustmentFaces(int vertexIndex, int level)
		{
			std::vector<std::vector<Face>> result;
			for (int iter = 0; iter <= level; iter++)
			{
				result.push_back(std::vector<Face>());
			}
			for (int iter=0; iter < levels[level].numberOfFaces; iter++)
			{
				if (levels[level].faces[iter].vertA == vertexIndex)
				{
					result[level].push_back(levels[level].faces[iter]);
				}
				if (levels[level].faces[iter].vertB == vertexIndex)
				{
					result[level].push_back(levels[level].faces[iter]);
				}
				if (levels[level].faces[iter].vertC == vertexIndex)
				{
					result[level].push_back(levels[level].faces[iter]);
				}
			}
			for (int iter = level - 1; iter >= 0; iter--)
			{
				for (int iter1 = 0; iter1 < result[iter + 1].size(); iter1++)
				{
					result[iter].push_back(levels[iter].faces[result[iter + 1][iter1].parentFace]);
				}
			}
			return result;
		}

		bool sameSide(const glm::vec3 A1, const glm::vec3 A2, const glm::vec3 testPoint, const glm::vec3 point)
		{
			glm::vec3 normal = glm::normalize(glm::cross(A1, A2 - A1));
			if (glm::dot(normal, point) != 0.0f)
			{
				return (glm::dot(normal, point)*glm::dot(normal, testPoint) > 0);
			}
			else {
				return true;
			}
		}

		bool testTriangle(const glm::vec3 A, const glm::vec3 B, const glm::vec3 C, const glm::vec3 point)
		{
			bool result = true;

			return sameSide(A, B, C, point) && sameSide(B, C, A, point) && sameSide(C, A, B, point);
		}


	private:
		void initLevel();
		Level* levels = nullptr;
		int _numberOfLevels;
		bool performSubdivision(Level* prevLevel, Level* nextLevel);
	};



}




