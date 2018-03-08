// OfflineRendering.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TexturedPlaneSampler.h"
#include "SubdivisionSphere.h"
#include "ColoredPlaneSampler.h"
#undef main
using namespace glm;
using namespace std;

int main(void)
{
	// some aliases for readability
	using Plane = TexturedPlaneSampler::Plane;
	using Ray = Sampler::Ray;
	
	// create Sampler
	Plane plane(vec3(0, 0, 10), vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0)); // mind the backface culling!
	const auto planeSampler = std::make_unique<ColoredPlaneSampler>(glm::vec3(0.f, 1.f, 0.0f), glm::vec3(1.0f, 0.f, 0.f), plane);

	// create demo ray(s) & sample
	Ray ray(vec3(0), vec3(0, 0, 1));
	auto color = planeSampler->sample(ray);
	ray = Ray(vec3(0), vec3(1,0,0));
	color = planeSampler->sample(ray);
	ray = Ray(vec3(0), vec3(0.1, 0,0.1));
	color = planeSampler->sample(ray);
	ray = Ray(vec3(10.1f), vec3(0.1, 0, 0.1));
	color = planeSampler->sample(ray);

	// use the spheres
	auto sphereData = make_unique<SubdivisionShpere::SubdivisionSphere>(5u);
	auto level0 = sphereData->getLevel(1u);
	auto colors = make_unique<vector<vec3>>();
	colors->reserve(level0.numberOfVertices * level0.numberOfVertices);
	for(auto positionVertexIterator = level0.vertices; positionVertexIterator != (level0.vertices + level0.numberOfVertices); ++positionVertexIterator)
	{
		auto& position = positionVertexIterator->position;
		auto l = length(position);
		for(auto rotationVertexIterator = level0.vertices; rotationVertexIterator != (level0.vertices + level0.numberOfVertices); ++rotationVertexIterator)
		{
			auto& rotation = rotationVertexIterator->position;
			colors->push_back(planeSampler->sample(Ray(position, position + rotation)));
		}
	}

	return 0;
}

