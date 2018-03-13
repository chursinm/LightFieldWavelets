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
	}

	shared_ptr<vector<vec3>> LightfieldLevel::rawData()
	{
		return mRawData;
	}

	// see https://gamedev.stackexchange.com/a/49370
	// Compute barycentric coordinates (u, v, w) for
	// point p with respect to triangle (a, b, c)
	// TODO move this to the subdivision project. precalculate and store v0, v1, d00, d01, d11 and invDenom while at it.
	void Barycentric(const vec3& p, const vec3& a, const vec3& b, const vec3& c, vec3& uvw)
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
	}

	vector<vec3> LightfieldLevel::snapshot(const vec3& cameraPositionInPositionSphereSpace) const
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
			const auto rawDataIndexA = i * levelData.numberOfVertices + facePtr->vertA;
			const auto rawDataIndexB = i * levelData.numberOfVertices + facePtr->vertB;
			const auto rawDataIndexC = i * levelData.numberOfVertices + facePtr->vertC;
			const auto rawDataA = mRawData->data()[rawDataIndexA];
			const auto rawDataB = mRawData->data()[rawDataIndexB];
			const auto rawDataC = mRawData->data()[rawDataIndexC];

			vec3 uvw(0.f);
			Barycentric(localRotation, facePtr->vertARef->position, facePtr->vertBRef->position, facePtr->vertCRef->position, uvw);

			//result.push_back(uvw);
			result.push_back(uvw.x * rawDataA + uvw.y * rawDataB + uvw.z * rawDataC);
		}

		return move(result);
	}
}
