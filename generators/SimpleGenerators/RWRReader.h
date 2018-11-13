#pragma once
#include "Ray.h"
#include <LightFieldData.h>
#include <SubdivisionSphere.h>



namespace Generator
{
	
	class RWRReader
	{
	public:
		RWRReader(const std::string& nameOfRWRFileIn, double bla) ;
		void readRWRFile();
		void projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData) ;
		const auto getNameOfRWRFile() { return nameOfRWRFile; };

	private:
		std::vector<Ray>* allRays;
		void accumilateRay(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData,const LightField::SphereLevel& level, Ray& ray, float invEnergy, float mult);		
		const std::string nameOfRWRFile="";
		double scaleLuminance=1.0f;
		
	};
}