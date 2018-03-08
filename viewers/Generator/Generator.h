#pragma once
#include "Sampler.h"
#include "SubdivisionSphere.h"

namespace Generator
{
	struct LightfieldData
	{
		// Global Metadata
		float minTriangleEdgeLength;
		float avgTriangleEdgeLength;
		float maxTriangleEdgeLength;
		unsigned int subdivisionLevel;
		unsigned int positionSampleCount;
		unsigned int rotationSampleCount; // == positionSampleCount, as subdivisionLevel (SL) = positionSL = rotationSL
		unsigned int totalSampleCount;
		glm::vec3 sphereCenter; // basically always 0,0,0

		// Per Sphere Data
		SubdivisionShpere::Face *faces;
		SubdivisionShpere::Vertex *vertices;
		std::vector<glm::vec3> normal; // eg normal[i] = normalize( (vertices+i)->position - sphereCenter )

		// Per Ray Data (Sphere x Sphere)
		std::vector<glm::vec3> color;
		std::vector<Sampler::Sampler::Ray> ray;
		std::vector<bool> hitMiss;
		std::vector<float> hitDepth;
		std::vector<glm::vec3> hitPosition;
	};
	class Generator
	{
	public:
		Generator();
		~Generator();
	};
}
