#include "stdafx.h"
#include "LightfieldLevel.h"
using namespace std;
using namespace glm;
namespace Generator
{
	LightfieldLevel::LightfieldLevel(std::shared_ptr<SubdivisionShpere::SubdivisionSphere> sphere, unsigned level,
		const Sampler::Sampler& sampler) : mSphere(sphere), mLevel(level), mRawData(make_shared<vector<vec3>>())
	{
		auto i = 0u;
		const auto levelData = mSphere->getLevel(mLevel);
		mRawData->reserve(levelData.numberOfVertices * levelData.numberOfVertices);
		for(auto positionVertexIterator = levelData.vertices; positionVertexIterator != (levelData.vertices + levelData.numberOfVertices); ++positionVertexIterator)
		{
			const auto& position = positionVertexIterator->position;
			for(auto rotationVertexIterator = levelData.vertices; rotationVertexIterator != (levelData.vertices + levelData.numberOfVertices); ++rotationVertexIterator)
			{
				const auto& rotation = rotationVertexIterator->position;
				mRawData->data()[i++] = sampler.sample(Sampler::Sampler::Ray(position, -rotation));
				//mRawData->push_back(sampler.sample(Sampler::Sampler::Ray(position, -rotation)));
			}
		}
		std::cout << "Next level would be " << levelData.numberOfVertices * levelData.numberOfVertices * sizeof(glm::vec3) << " byte" << std::endl;
	}

	shared_ptr<vector<vec3>> LightfieldLevel::rawData()
	{
		return mRawData;
	}

	vector<vec3> LightfieldLevel::snapshot(vec3 cameraPositionInPositionSphereSpace)
	{
		const auto levelData = mSphere->getLevel(mLevel);

		vector<vec3> result;
		result.reserve(levelData.numberOfVertices);
		for(auto i = 0u; i < levelData.numberOfVertices; ++i)
		{
			// Calculate faces for each position
			const auto localRotation = normalize(cameraPositionInPositionSphereSpace - levelData.vertices[i].position);
			const auto faceIndex = mSphere->vectorToFaceIndex(localRotation, mLevel);
			const auto facePtr = levelData.faces + faceIndex;

			// Interpolate the color using the face & barycentric coordinates
			// TODO implement barycentric coordinates
			const auto rawDataIndex = i * levelData.numberOfVertices + facePtr->vertA; // and vertB/vertC -> interpolate .. see TODU
			result.push_back(mRawData->data()[rawDataIndex]);
		}

		return move(result);
	}
}
