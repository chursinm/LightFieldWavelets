#pragma once
#include "Ray.h"



namespace Generator
{
	class RWRReader
	{
	public:
		explicit RWRReader(const std::string nameOfRWRFile);	
		void projectRaysToSphere(const LightField::SubdivisionSphere& subsphere, LightField::LightFieldData& lightFieldData);
	private:
		std::vector<Ray> mAllRays;
		
		
	};
}