// OfflineRendering.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "TexturedPlaneSampler.h"
#include "SubdivisionSphere.h"
#include "ColoredPlaneSampler.h"
#include "SetSampler.h"
//#include "LightfieldLevel.h"
#include "CheckerboardSampler.h"
#include "LightFieldСontainer.h"

using namespace glm;
using namespace std;
using namespace Generator::Sampler;

bool floatEqual(const float& a, const float& b)
{
	constexpr auto epsilon = 0.00001f;
	return abs(a - b) < epsilon;
}

void zeigeSeitenverhaeltnisse(LightField::SubdivisionSphere* sphereData)
{
	for(auto level = 0u; level < sphereData->getNumberOfLevels(); ++level)
	{
		auto levelData = sphereData->getLevel(level);
		auto gleichseitige = 0u;
		auto gleichschenkelige = 0u;
		auto allgemeine = 0u;
		for(auto positionFacesIterator: levelData.getFaces())
		{
			auto& vert01 = positionFacesIterator.vertices[0]->pos;
			auto& vert02 = positionFacesIterator.vertices[1]->pos;
			auto& vert03 = positionFacesIterator.vertices[2]->pos;
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

void zeigePolarkoordinaten(LightField::SubdivisionSphere* sphereData)
{
	constexpr auto pi = 3.1415926535897932384626433832795f;
	for(auto level = 0u; level < sphereData->getNumberOfLevels(); ++level)
	{
		auto levelData = sphereData->getLevel(level);
		std::cout << "Level " << level << std::endl;
		for(auto vertexIterator : levelData.getVertices())
		{
			auto& vertex = vertexIterator.pos;
			vec3 test = normalize(vertex);
			// Rotation Y Axis (A)
			vec2 yRotVert = test.xz; //normalize(vertex.xz);
			float latitude = atan(yRotVert.x, yRotVert.y) * 180.f / pi;
			// Rotation X Axis (b)
			vec2 xRotVert = test.yz; //normalize(vertex.yz);
			float longitude = acos(xRotVert.x) * 180.f / pi;

			std::cout << "lat: " << latitude << ", long: " << longitude << std::endl;
		}
	}
}

void testRays(const std::string& fileName)
{
	HellaRayExtractor::RayExtractor * rwrdata = new HellaRayExtractor::RayExtractor((char*)fileName.c_str());

	// Show which data is provided with each ray
	// Attention: non available data is initialized with default values
	HellaRayExtractor::RayInfo info = rwrdata->getRayInfo();
	cout << "Available ray components:" << endl << endl;
	if (info.has_startPoint)            cout << "  startPoint" << endl;
	if (info.has_direction)             cout << "  direction" << endl;
	if (info.has_intensity)             cout << "  intensity" << endl;
	if (info.has_CIEx && info.has_CIEy) cout << "  CIEx and CIEy" << endl;
	if (info.has_wavelength)            cout << "  wavelength" << endl;
	if (info.has_layer)                 cout << "  layer" << endl << endl;;

	// extract all rays 
	unsigned __int64 numRays = rwrdata->getTotalNumberOfRays();
	unsigned __int64 stepSize = numRays / 100;
	cout << "Number of rays in RWR file : " << numRays << endl;
	cout << "Press any key to continue";
	cin.get();
	HellaRayExtractor::Ray nextRay;
	unsigned __int64 rayCount = 0;
	while (rwrdata->getNextRay(nextRay)) {

		if (rayCount % stepSize == 0) {
			cout.precision(2);
			cout << 100.0*(double)rayCount / (double)numRays << " %\t:("
				<< nextRay.startPoint[0] << ",\t"
				<< nextRay.startPoint[1] << ",\t"
				<< nextRay.startPoint[2] << "),  \t("
				<< nextRay.direction[0] << ",\t"
				<< nextRay.direction[1] << ",\t"
				<< nextRay.direction[2] << "),\t"
				<< nextRay.intensity << ",\t("
				<< nextRay.CIEx << " " << nextRay.CIEy << "),\t"
				<< nextRay.layer << endl;
		}
		rayCount++;
	};

}

int main(int argc, char **argv)
{
	/*auto sphereLevelCount = 1u;
	if(argc == 2)
	{
		stringstream(argv[1]) >> sphereLevelCount;
	}

	// some aliases for readability
	using Plane = TexturedPlaneSampler::Plane;
	
	// create Sampler
	Plane plane(vec3(0, 0, 10), vec3(0, 0, -1), vec3(0, 1, 0), vec3(1, 0, 0)); // mind the backface culling!
	const auto planeSampler = make_shared<CheckerboardSampler>(0.f, glm::vec3(1.0f, 0.f, 0.f), plane);
	Plane plane2(vec3(0, 10, 0), vec3(0, -1, 0), vec3(0, 0, 1), vec3(1, 0, 0)); // mind the backface culling!
	const auto plane2Sampler = make_shared<CheckerboardSampler>(0.f, glm::vec3(0.5f, 0.f, 0.f), plane2);
	const auto setSampler = make_unique<SetSampler>(vector<shared_ptr<Sampler>> { planeSampler, plane2Sampler }, glm::vec3(0.2f,0.f,0.f));

	auto sphereData = std::make_shared<LightField::SubdivisionSphere>(sphereLevelCount);

	for(auto levelIndex = 0u; levelIndex < sphereData->getNumberOfLevels(); ++levelIndex)
	{
		auto levelData = sphereData->getLevel(levelIndex);
		for(auto facesIterator : levelData.getFaces())
		{
			auto centralVertex = normalize(	facesIterator.vertices[0]->pos + 
											facesIterator.vertices[1]->pos +
											facesIterator.vertices[2]->pos 	);
			auto halfCentralVertex = normalize(centralVertex + facesIterator.vertices[1]->pos);
			auto faceIndex = sphereData->vectorToFaceIndex(halfCentralVertex, levelIndex);
			if(faceIndex != facesIterator.index)
				std::cout << "break";
		}
	}

	Generator::LightfieldLevel lfl(sphereData, 0, *setSampler);
	auto rawData = lfl.rawData();
	auto cam0Data = lfl.snapshot(vec3(0,0,0));*/

	std::cout << "test ray file" << std::endl;
	//testRays("c:/temp/test.rwr");

	
	auto subsphere = std::make_shared<LightField::SubdivisionSphere>(3);
	auto  rwrReader = std::make_shared<Generator::RWRReader>("c:/temp/test.rwr", 1.0);

	Generator::LightFieldСontainer lfg(subsphere, glm::vec3(0.0f), rwrReader);

	return 0;
}

