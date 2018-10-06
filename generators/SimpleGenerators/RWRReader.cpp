#include "stdafx.h"
#include "RWRReader.h"
#include <glm/gtc/random.hpp>


namespace Generator
{
	RWRReader::RWRReader(const std::string nameOfRWRFileIn)
	{
		HellaRayExtractor::RayExtractor * rwrdata = new HellaRayExtractor::RayExtractor((char*)nameOfRWRFileIn.c_str());
		HellaRayExtractor::RayInfo info = rwrdata->getRayInfo();
		unsigned __int64 numRays = rwrdata->getTotalNumberOfRays();
		unsigned __int64 stepSize = numRays / 100;
		HellaRayExtractor::Ray nextRay;
		unsigned __int64 rayCount = 0;
		while (rwrdata->getNextRay(nextRay)) {
			glm::vec3 origin(nextRay.startPoint[0], nextRay.startPoint[1], nextRay.startPoint[2]);
			glm::vec3 direction(nextRay.direction[0], nextRay.direction[1], nextRay.direction[2]);
			allRays.push_back(Ray(origin, direction, nextRay.intensity));
		}
		rayCount++;
	};
	
	void RWRReader::projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData)
	{		
		for (const auto& level : subsphere->getLevels())
		{
			int numberOfRays = 100000;
			float numberOfRaysInv = 0.0001f* (level.getNumberOfFaces()*level.getNumberOfFaces()) / (numberOfRays);
			for (int i = 0; i < numberOfRays; i++)
			{
				auto diskPos = glm::diskRand(0.3f);
				auto diskRot = glm::diskRand(0.3f);
				Ray ray(glm::normalize(glm::vec3(diskPos.x, diskPos.y, -1.0f)), glm::normalize(glm::vec3(diskRot.x, diskRot.y,1.0f)), 1);

				int faceIndPos = subsphere->vectorToFaceIndex(ray.mOrigin, level.getIndex());
				int faceIndRot = subsphere->vectorToFaceIndex(ray.mDirection, level.getIndex());

				auto facePos = subsphere->getLevel(level.getIndex()).getFace(faceIndPos);
				auto faceRot = subsphere->getLevel(level.getIndex()).getFace(faceIndRot);

				glm::vec3 uvwPos, uvwRot;
				LightField::LightFieldData::Barycentric(ray.mOrigin, facePos.vertices[0]->pos, facePos.vertices[1]->pos, facePos.vertices[2]->pos, uvwPos);
				LightField::LightFieldData::Barycentric(ray.mDirection, faceRot.vertices[0]->pos, faceRot.vertices[1]->pos, faceRot.vertices[2]->pos, uvwRot);

				for (int k = 0; k < 3; k++)
				{
					for (int l = 0; l < 3; l++)
					{

						lightFieldData.getLevelMatrix(level.getIndex())->addValue(glm::vec3(1.0f, 1.0f, 1.0f)*uvwPos[k]* uvwRot[l]* numberOfRaysInv,
							facePos.vertices[k]->index,
							faceRot.vertices[l]->index
						);

					}
				}
			}
		}
	}
}
