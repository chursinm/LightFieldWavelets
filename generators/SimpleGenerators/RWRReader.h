#pragma once
#include "Ray.h"



namespace Generator
{
	class RWRReader
	{
	public:
		explicit RWRReader(const std::string nameOfRWRFile) {};
		void projectRaysToSphere(std::shared_ptr<LightField::SubdivisionSphere> subsphere, LightField::LightFieldData& lightFieldData) {};
	private:
		std::vector<Ray> allRays;
		
		
	};
}