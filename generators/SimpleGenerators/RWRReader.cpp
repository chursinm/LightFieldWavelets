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
		allRays->reserve(numRays);
		while (rwrdata->getNextRay(nextRay)) {
			//if (rayCount % 1 == 0)
			{
				glm::vec3 origin(nextRay.startPoint[0], nextRay.startPoint[1], nextRay.startPoint[2]);
				glm::vec3 direction(nextRay.direction[0], nextRay.direction[1], nextRay.direction[2]);
				allRays->push_back(Ray(origin, direction, nextRay.intensity, nextRay.CIEx, nextRay.CIEy, nextRay.layer));
				//allRays->at(allRays->size() - 1).getRGB();
				if (rayCount%10000000 == 0)
				std::cout << "read " << rayCount/1 << " rays" << std::endl ;
			}
			rayCount++;
		}
		
	};
	


	void RWRReader::projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData)
	{	
		const float scale = 1.0f / 170.0f;
		//const float scale = 1.0f / 30.0f;
		//for (const auto& level : subsphere->getLevels())
		const auto& level = subsphere->getLevel(subsphere->getNumberOfLevels()-1);
		{
			std::cout << "extract rays to level " << level.getIndex() << std::endl;
			int numberOfRays = allRays->size();
			//numberOfRays = 10000;
			float numberOfRaysInv = 6.0f*(level.getNumberOfFaces()) / numberOfRays;
			for (int i = 0; i < numberOfRays; i++)
			{
				if (i % 1000000 == 0)
				{
					std::cout << "project rays number: " << i << std::endl;
				}
				
				

				//auto diskPos = glm::diskRand(0.3f);S
				//auto diskRot = glm::diskRand(0.3f);


				//glm::vec3 spherePos = glm::sphericalRand(0.1f);
				//glm::vec3 spherePos = glm::vec3(0.0f, 0.0f, 0.0f);
				//glm::vec3 spherePos = glm::vec3(0.0f, 0.0f, 0.0f);
				//glm::vec3 sphereRot =  glm::normalize(glm::vec3(0.0f, 0.0f, 1.0f));
				//glm::vec3 sphereRot = sphericalRand(1.0f);
				//Ray rayIn(spherePos, sphereRot, 1, 0.5, 0.5, 0);
				//accumilateRay(subsphere, lightFieldData, level, rayIn, numberOfRaysInv, 1.0f);
				//Ray rayOut(spherePos, -sphereRot, 1, 0.5, 0.5, 0);
				//accumilateRay(subsphere, lightFieldData, level, rayOut, numberOfRaysInv, -1.0f);

				
				Ray& ray = allRays->at(i);

				Ray rayIn(scale*ray.mOrigin, ray.mDirection, ray.intensity, ray.CIEx, ray.CIEy, ray.layer);
				
				accumilateRay(subsphere, lightFieldData, level, rayIn, numberOfRaysInv, 1.0f);
				
				Ray rayOut(scale*ray.mOrigin, -ray.mDirection, ray.intensity, ray.CIEx, ray.CIEy, ray.layer);
				
				accumilateRay(subsphere, lightFieldData, level, rayOut, numberOfRaysInv, -1.0f);

			}
		}
		delete allRays;
	}
	void RWRReader::accumilateRay(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData & lightFieldData, const LightField::SphereLevel& level, Ray & ray, float invEnergy, float mult)
	{
		glm::vec3 resPos;
		glm::vec3 resNorm;
		auto answ = glm::intersectRaySphere(ray.mOrigin, -ray.mDirection, glm::vec3(0.0f), 1.0f, resPos, resNorm);
		if (answ)
		{
			int faceIndPos = subsphere->vectorToFaceIndex(resPos, level.getIndex());
			int faceIndRot = subsphere->vectorToFaceIndex(mult*ray.mDirection, level.getIndex());

			auto facePos = subsphere->getLevel(level.getIndex()).getFace(faceIndPos);
			auto faceRot = subsphere->getLevel(level.getIndex()).getFace(faceIndRot);

			glm::vec3 uvwPos, uvwRot;
			LightField::LightFieldData::Barycentric(resPos, facePos.vertices[0]->pos, facePos.vertices[1]->pos, facePos.vertices[2]->pos, uvwPos);
			LightField::LightFieldData::Barycentric(mult*ray.mDirection, faceRot.vertices[0]->pos, faceRot.vertices[1]->pos, faceRot.vertices[2]->pos, uvwRot);




			for (int k = 0; k < 3; k++)
			{
				for (int l = 0; l < 3; l++)
				{					
					lightFieldData.getLevelMatrix(level.getIndex())->addValue(ray.getRGB() *uvwPos[k] * uvwRot[l] * invEnergy*(float)(ray.intensity),
						facePos.vertices[k]->index,
						faceRot.vertices[l]->index
					);

				}
			}

		}

	}
}
