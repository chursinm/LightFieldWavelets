#include "stdafx.h"
#include "RWRReader.h"
#include <glm/gtc/random.hpp>
#include <glm/gtx/intersect.hpp>
#include <iostream>


namespace Generator
{
	RWRReader::RWRReader(const std::string nameOfRWRFileIn)
	{
		allRays = new std::vector<Ray>();
		HellaRayExtractor::RayExtractor * rwrdata = new HellaRayExtractor::RayExtractor((char*)nameOfRWRFileIn.c_str());
		HellaRayExtractor::RayInfo info = rwrdata->getRayInfo();
		unsigned __int64 numRays = rwrdata->getTotalNumberOfRays();
		unsigned __int64 stepSize = numRays / 100;
		HellaRayExtractor::Ray nextRay;
		unsigned __int64 rayCount = 0;
		while (rwrdata->getNextRay(nextRay)) {
			if (rayCount % 1 == 0)
			{
				glm::vec3 origin(nextRay.startPoint[0], nextRay.startPoint[1], nextRay.startPoint[2]);
				glm::vec3 direction(nextRay.direction[0], nextRay.direction[1], nextRay.direction[2]);
				allRays->push_back(Ray(origin, direction, nextRay.intensity));
				if (rayCount%10000000 == 0)
				std::cout << "read " << rayCount/1 << " rays" << std::endl ;
			}
			rayCount++;
		}
		
	};
	
	void RWRReader::projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData)
	{	
		for (const auto& level : subsphere->getLevels())
		{
			std::cout << "extract rays to level " << level.getIndex() << std::endl;
			int numberOfRays = allRays->size();
			
			for (int i = 0; i < numberOfRays; i++)
			{

				float numberOfRaysInv = 10.f*(level.getNumberOfFaces()) / numberOfRays;

				//auto diskPos = glm::diskRand(0.3f);
				//auto diskRot = glm::diskRand(0.3f);


				//glm::vec3 spherePos = glm::sphericalRand(0.2f);
				//glm::vec3 sphereRot = glm::sphericalRand(1.0f);

				Ray ray(allRays->at(i).mOrigin/75.0f, -allRays->at(i).mDirection, allRays->at(i).intensity/1.0f);

				glm::vec3 resPos;
				glm::vec3 resNorm;
				auto answ = glm::intersectRaySphere(ray.mOrigin, -ray.mDirection, glm::vec3(0.0f), 1.0f, resPos, resNorm);

				if (answ)
				{
					int faceIndPos = subsphere->vectorToFaceIndex(resPos, level.getIndex());
					int faceIndRot = subsphere->vectorToFaceIndex(ray.mDirection, level.getIndex());

					auto facePos = subsphere->getLevel(level.getIndex()).getFace(faceIndPos);
					auto faceRot = subsphere->getLevel(level.getIndex()).getFace(faceIndRot);

					glm::vec3 uvwPos, uvwRot;
					LightField::LightFieldData::Barycentric(resPos, facePos.vertices[0]->pos, facePos.vertices[1]->pos, facePos.vertices[2]->pos, uvwPos);
					LightField::LightFieldData::Barycentric(ray.mDirection, faceRot.vertices[0]->pos, faceRot.vertices[1]->pos, faceRot.vertices[2]->pos, uvwRot);




					for (int k = 0; k < 3; k++)
					{
						for (int l = 0; l < 3; l++)
						{

							lightFieldData.getLevelMatrix(level.getIndex())->addValue(glm::vec3(1.0f, 0.0f, 0.0f)*uvwPos[k] * uvwRot[l] * numberOfRaysInv*(float)(ray.intensity),
								facePos.vertices[k]->index,
								faceRot.vertices[l]->index
							);

						}
					}

				}

			}
		}
		delete allRays;
	}
}
