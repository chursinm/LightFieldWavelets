#include "stdafx.h"
#include "SubdivisionSphere.h"
#include "SphereVertex.h"



namespace LightField
{
	SubdivisionSphere::SubdivisionSphere(int numberOfLevels)
	{
		if (numberOfLevels > 0)
		{
			levels.emplace_back();
		}		
		for (int i = 1; i < numberOfLevels; i++)
		{			
			levels.emplace_back(&levels.back());
		}
	}
	int SubdivisionSphere::vectorToFaceIndex(const glm::vec3& vect, int maxLevelInd)
	{
		SphereFace* face = nullptr;
		
		
		for (auto& testFace:levels[0].faces)
		{
			testFace.vertices[0]->pos;

			if (testTriangle(
				testFace.vertices[0]->pos,
				testFace.vertices[1]->pos,
				testFace.vertices[2]->pos,
					vect))
			{
				face = &testFace;
				break;
			}
		}


		for (int iter = 1; iter <= maxLevelInd; iter++)
		{		
			if (sameSide(face->childFaces[0]->vertices[1]->pos, face->childFaces[0]->vertices[2]->pos, face->childFaces[0]->vertices[0]->pos, vect))
			{
				face = face->childFaces[0];
				continue;
			}
			if (sameSide(face->childFaces[1]->vertices[1]->pos, face->childFaces[1]->vertices[2]->pos, face->childFaces[1]->vertices[0]->pos, vect))
			{
				face = face->childFaces[1];
				continue;
			}
			if (sameSide(face->childFaces[2]->vertices[1]->pos, face->childFaces[2]->vertices[2]->pos, face->childFaces[2]->vertices[0]->pos, vect))
			{
				face = face->childFaces[2];
				continue;
			}



			face = face->childFaces[3];

		}

		return face->index;

	}


	bool SubdivisionSphere::sameSide(const glm::vec3 A1, const glm::vec3 A2, const glm::vec3 testPoint, const glm::vec3 point)
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

	bool SubdivisionSphere::testTriangle(const glm::vec3 A, const glm::vec3 B, const glm::vec3 C, const glm::vec3 point)
	{
		bool result = true;

		return sameSide(A, B, C, point) && sameSide(B, C, A, point) && sameSide(C, A, B, point);
	}
}