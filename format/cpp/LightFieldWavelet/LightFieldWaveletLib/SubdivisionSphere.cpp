#include "stdafx.h"
#include "SubdivisionSphere.h"
#include "SphereVertex.h"



namespace LightField
{
	SubdivisionSphere::SubdivisionSphere(int numberOfLevels)
	{
		levels.emplace_back();
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


		/*for (int iter = 1; iter <= level; iter++)
		{
			if (sameSide(face->childFaceARef->vertBRef->position, face->childFaceARef->vertCRef->position, face->childFaceARef->vertARef->position, vect))
			{
				face = face->childFaceARef;
				continue;
			}
			if (sameSide(face->childFaceBRef->vertBRef->position, face->childFaceBRef->vertCRef->position, face->childFaceBRef->vertARef->position, vect))
			{
				face = face->childFaceBRef;
				continue;
			}
			if (sameSide(face->childFaceCRef->vertBRef->position, face->childFaceCRef->vertCRef->position, face->childFaceCRef->vertARef->position, vect))
			{
				face = face->childFaceCRef;
				continue;
			}
			face = face->childFaceDRef;

		}*/

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