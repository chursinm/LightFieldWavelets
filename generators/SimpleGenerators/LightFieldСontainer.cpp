#include "stdafx.h"
#include "LightFieldСontainer.h"
#include <iostream>
#include <fstream>
#include <iterator>
#include <stdio.h>


using namespace glm;

Generator::LightFieldСontainer::LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, const glm::vec3& spherePosIn, std::shared_ptr<Sampler::Sampler> samplerIn):
	subdivisionSphere(sphereIn),
	sampler(samplerIn),
	lightFieldData(subdivisionSphere),
	spherePos(spherePosIn)
{
	for (const auto& level : subdivisionSphere->getLevels())
	{
		for (const auto& vPos : level.getVertices())
		{
			for (const auto& vRot : level.getVertices())
			{
				const auto& rotation = vRot.pos ;
				const auto& position = vPos.pos;
				const Ray ray(position+spherePos, -rotation);
				lightFieldData.getLevelMatrix(level.getIndex())->setValue(sampler->sample(ray),vPos.index, vRot.index);
			}
		}
	}
}

Generator::LightFieldСontainer::LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, const glm::vec3& spherePosIn,  std::shared_ptr<Generator::RWRReader> rwrReaderIn):
	subdivisionSphere(sphereIn),
	rwrReader(rwrReaderIn),
	lightFieldData(subdivisionSphere),
	spherePos(spherePosIn)
{
	
	std::string nameOfRwrFile = rwrReader->getNameOfRWRFile();	
	std::string nameOfMtrFile = nameOfRwrFile;
	size_t start_pos = nameOfRwrFile.find(".rwr");
	if (start_pos == std::string::npos) nameOfMtrFile = "c:/tmp.mtr";
	nameOfMtrFile.replace(start_pos, std::string(".rwr").length(),  std::string(".mtr"));
	std::cout << "name of mtr file: " << nameOfMtrFile << std::endl;


	
	if (!tryToReadMtrFile(nameOfMtrFile))
	{
		rwrReader->readRWRFile();
		rwrReader->projectRaysToSphere(subdivisionSphere, lightFieldData);
		writeMrtFile(nameOfMtrFile);
	}
	
}


bool Generator::LightFieldСontainer::tryToReadMtrFile(const std::string fileName)
{
	std::ifstream file (fileName, std::ios::binary);
	if (!file.good())
		return false;
	auto data = lightFieldData.getLevelMatrix(subdivisionSphere->getNumberOfLevels() - 1)->getData();
	vec3 temp;
	int numberOfElements = data->size();
	
	data->clear();
	while (file.read((char*)&temp, sizeof(temp)))
		data->push_back(temp);
	file.close();
	if (numberOfElements != data->size())
	{
		std::cout << "error in reading file. delete temp .mtr file" << std::endl;
		std::remove(fileName.c_str());
		throw "error reading file";
		return false;
	}
	return true;
}

bool Generator::LightFieldСontainer::writeMrtFile(const std::string fileName)
{
	std::ofstream file(fileName, std::ios::binary);
	auto data = lightFieldData.getLevelMatrix(subdivisionSphere->getNumberOfLevels()-1)->getData();
	//data->at(2).y = 12.0;
	for (const auto& element : *data)
	{
		file.write((char*)&element, sizeof(vec3));
	}
	//auto size = data->size() * sizeof(vec3);
	//file.write((char*)data, size);
	
	file.close();
	return true;
}


// see https://gamedev.stackexchange.com/a/49370
// Compute barycentric coordinates (u, v, w) for
// point p with respect to triangle (a, b, c)
// TODO move this to the subdivision project. precalculate and store v0, v1, d00, d01, d11 and invDenom while at it.
/*void Barycentric(const vec3& p, const vec3& a, const vec3& b, const vec3& c, vec3& uvw)
{
	const auto v0 = b - a, v1 = c - a, v2 = p - a;
	const auto d00 = dot(v0, v0);
	const auto d01 = dot(v0, v1);
	const auto d11 = dot(v1, v1);
	const auto d20 = dot(v2, v0);
	const auto d21 = dot(v2, v1);
	const auto invDenom = 1.0f / (d00 * d11 - d01 * d01);
	uvw.y = (d11 * d20 - d01 * d21) * invDenom;
	uvw.z = (d00 * d21 - d01 * d20) * invDenom;
	uvw.x = 1.0f - uvw.y - uvw.z;
}*/

std::vector<vec3> Generator::LightFieldСontainer::snapshot(const glm::vec3 & cameraPositionInPositionSphereSpace,int levelInd) const
{
	std:: vector<glm::vec3> result;
	result.reserve(subdivisionSphere->getLevel(levelInd).getNumberOfVertices());
	for (const auto& vPos : subdivisionSphere->getLevel(levelInd).getVertices())
	{
		const auto localRotation = glm::normalize(cameraPositionInPositionSphereSpace - vPos.pos-spherePos);
		const auto faceIndex = subdivisionSphere->vectorToFaceIndex(localRotation, levelInd);
		const auto face = subdivisionSphere->getLevel(levelInd).getFace(faceIndex);
		vec3 uvw(0.f);
		LightField::LightFieldData::Barycentric(localRotation, face.vertices[0]->pos, face.vertices[1]->pos, face.vertices[2]->pos, uvw);
		
		
		result.push_back(
							uvw.x * lightFieldData.getLevelMatrix(levelInd)->getValue(vPos.index, face.vertices[0]->index)	+
							uvw.y * lightFieldData.getLevelMatrix(levelInd)->getValue(vPos.index, face.vertices[1]->index)	+
							uvw.z * lightFieldData.getLevelMatrix(levelInd)->getValue(vPos.index, face.vertices[2]->index));
	}
	return std::move(result);
}

