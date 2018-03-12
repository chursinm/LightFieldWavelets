#pragma once
#include "Sampler.h"
#include "SubdivisionSphere.h"

namespace Generator
{
	struct LightfieldData
	{
		// Per Level
		float minTriangleEdgeLength;
		float avgTriangleEdgeLength;
		float maxTriangleEdgeLength;
		unsigned int subdivisionLevel;
		unsigned int positionSampleCount;
		unsigned int rotationSampleCount; // == positionSampleCount, as subdivisionLevel (SL) = positionSL = rotationSL
		unsigned int totalSampleCount;

		// Per Sphere
		glm::vec3 sphereCenter; // basically always 0,0,0
		float sphereRadius; // should be 1, closer to 0.98 in reality
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
