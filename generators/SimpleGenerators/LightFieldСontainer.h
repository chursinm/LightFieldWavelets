#pragma once
//#include "LightfieldLevel.h"
#include "Sampler.h"
#include "RWRReader.h"

namespace Generator
{
	class LightFieldСontainer
	{
	public:
		explicit LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, const glm::vec3& spherePosIn, std::shared_ptr<Sampler::Sampler> samplerIn);


		explicit LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, const glm::vec3& spherePosIn, std::shared_ptr<Generator::RWRReader> rwrReaderIn );

		std::vector<glm::vec3> snapshot(const glm::vec3& cameraPositionInPositionSphereSpace, int levelInd) const;

		LightField::LightFieldData& getLightFieldData() { return lightFieldData; }

		~LightFieldСontainer() = default;		
	private:	
		std::shared_ptr<LightField::SubdivisionSphere>	subdivistionSphere;
		std::shared_ptr<Sampler::Sampler>				sampler;
		std::shared_ptr<Generator::RWRReader>			rwrReader;
		LightField::LightFieldData						lightFieldData;
		const glm::vec3									spherePos;
		
		
	};
}
