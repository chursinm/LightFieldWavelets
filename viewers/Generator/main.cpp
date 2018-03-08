// OfflineRendering.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TexturedPlaneSampler.h"
#include "SubdivisionSphere.h"
#include "ColoredPlaneSampler.h"
#include "SetSampler.h"
using namespace glm;
using namespace std;
using namespace Generator::Sampler;

bool floatEqual(const float& a, const float& b)
{
	constexpr auto epsilon = 0.00001f;
	return abs(a - b) < epsilon;
}

void zeigeSeitenverhaeltnisse(SubdivisionShpere::SubdivisionSphere* sphereData)
{
	for(auto level = 0u; level < sphereData->getNumberOfLevels(); ++level)
	{
		auto levelData = sphereData->getLevel(level);
		auto gleichseitige = 0u;
		auto gleichschenkelige = 0u;
		auto allgemeine = 0u;
		for(auto positionFacesIterator = levelData.faces; positionFacesIterator != (levelData.faces + levelData.numberOfFaces); ++positionFacesIterator)
		{
			auto& vert01 = positionFacesIterator->vertARef->position;
			auto& vert02 = positionFacesIterator->vertBRef->position;
			auto& vert03 = positionFacesIterator->vertCRef->position;
			auto d1 = distance(vert01, vert02);
			auto d2 = distance(vert01, vert03);
			auto d3 = distance(vert02, vert03);

			if(floatEqual(d1, d2) && floatEqual(d2, d3))
			{
				gleichseitige++;
			}
			else if(floatEqual(d1, d2) || floatEqual(d2, d3) || floatEqual(d1, d3))
			{
				gleichschenkelige++;
			}
			else
			{
				allgemeine++;
			}
		}
		std::cout << "Level " << level << " hat " << gleichseitige << " gleichseitige, " << gleichschenkelige << " gleichschenkelige und " << allgemeine << " allgemeine Dreiecke (vllt auch rechtwinkelige)\n";
	}
}

int main(int argc, char **argv)
{
	auto sphereLevelCount = 6u;
	if(argc == 2)
	{
		stringstream(argv[1]) >> sphereLevelCount;
	}

	// some aliases for readability
	using Plane = TexturedPlaneSampler::Plane;
	using Ray = Sampler::Ray;
	
	// create Sampler
	Plane plane(vec3(0, 0, 10), vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0)); // mind the backface culling!
	const auto planeSampler = make_shared<ColoredPlaneSampler>(glm::vec3(0.f, 1.f, 0.0f), glm::vec3(1.0f, 0.f, 0.f), plane);
	Plane plane2(vec3(0, 10, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(1, 0, 0)); // mind the backface culling!
	const auto plane2Sampler = make_shared<ColoredPlaneSampler>(glm::vec3(0.f, 0.5f, 0.0f), glm::vec3(0.5f, 0.f, 0.f), plane2);
	const auto setSampler = make_unique<SetSampler>(vector<shared_ptr<Sampler>> { planeSampler, plane2Sampler }, glm::vec3(0.2f,0.f,0.f));

	// use the spheres
	auto sphereData = make_unique<SubdivisionShpere::SubdivisionSphere>(sphereLevelCount);
	auto levelData = sphereData->getLevel(0u);
	auto colors = make_unique<vector<vec3>>();
	colors->reserve(levelData.numberOfVertices * levelData.numberOfVertices);
	for(auto positionVertexIterator = levelData.vertices; positionVertexIterator != (levelData.vertices + levelData.numberOfVertices); ++positionVertexIterator)
	{
		auto& position = positionVertexIterator->position;
		for(auto rotationVertexIterator = levelData.vertices; rotationVertexIterator != (levelData.vertices + levelData.numberOfVertices); ++rotationVertexIterator)
		{
			auto& rotation = rotationVertexIterator->position;
			colors->push_back(setSampler->sample(Ray(position, -rotation)));
		}
	}
	zeigeSeitenverhaeltnisse(sphereData.get());

	return 0;
}

