#pragma once
#include "Ray.h"
#include <LightFieldData.h>
#include <SubdivisionSphere.h>



namespace Generator
{
	
	class RWRReader
	{
	public:
		explicit RWRReader(const std::string nameOfRWRFileIn) ;
		void projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData) ;		
	private:
		std::vector<Ray>* allRays;
		void accumilateRay(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData,const LightField::SphereLevel& level, Ray& ray, float invEnergy, float mult);
		
		
	};
}