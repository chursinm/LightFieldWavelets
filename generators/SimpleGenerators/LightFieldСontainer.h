#pragma once
//#include "LightfieldLevel.h"
#include "Sampler.h"
#include "RWRReader.h"

namespace Generator
{
	class LightFieldСontainer
	{
	public:
		explicit LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, std::shared_ptr<Sampler::Sampler> samplerIn);

		explicit LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, std::shared_ptr<Generator::RWRReader> rwrReaderIn );

		~LightFieldСontainer() = default;		
	private:	
		std::shared_ptr<LightField::SubdivisionSphere>	subdivistionSphere;
		std::shared_ptr<Sampler::Sampler>				sampler;
		std::shared_ptr<Generator::RWRReader>			rwrReader;
		LightField::LightFieldData						lightFieldData;
		
		
	};
}
