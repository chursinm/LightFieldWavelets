#include "stdafx.h"
#include "LightfieldLevel.h"

using namespace std;
using namespace glm;
namespace Generator
{
	LightfieldLevel::LightfieldLevel(std::shared_ptr<LightField::SubdivisionSphere> sphere, unsigned level,
		const Sampler::Sampler& sampler) : mSphere(sphere), mRawData(make_shared<vector<vec3>>()), mPositionSphereLevel(level), mRotationSphereLevel(level) // TODO p!=r sphere
	{
		auto i = 0ull;
		const auto positionLevelData = mSphere->getLevel(mPositionSphereLevel);
		const auto rotationLevelData = mSphere->getLevel(mRotationSphereLevel);
		const auto allocationSize = static_cast<vector<vec3>::size_type>(positionLevelData.getNumberOfVertices()) * static_cast<vector<vec3>::size_type>(rotationLevelData.getNumberOfVertices());
		std::cout << "trying to allocate " << allocationSize << " elements (" << allocationSize * sizeof(vec3) / 1073741824ull << " GByte)\n";
		mRawData->reserve(allocationSize);
		for(auto positionVertexIterator : positionLevelData.getVertices())
		{
			const auto& position = positionVertexIterator.pos;
			for(auto rotationVertexIterator : positionLevelData.getVertices())
			{
				const auto& rotation = rotationVertexIterator.pos;
				const Ray ray(position, -rotation, nullptr);
				mRawData->push_back(sampler.sample(ray));
			}
		}
	}

	shared_ptr<vector<vec3>> LightfieldLevel::rawData() const
	{
		return mRawData;
	}

	unsigned short LightfieldLevel::positionSphereLevel() const
	{
		return mPositionSphereLevel;
	}

	unsigned short LightfieldLevel::rotationSphereLevel() const
	{
		return mRotationSphereLevel;
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
		const auto positionLevelData = mSphere->getLevel(mPositionSphereLevel);
		const auto rotationLevelData = mSphere->getLevel(mRotationSphereLevel);

		vector<vec3> result;
		result.reserve(positionLevelData.getNumberOfVertices());
		for(auto i = 0ull; i < positionLevelData.getNumberOfVertices(); ++i)
		{
			// Calculate faces for each position
			const auto localRotation = normalize(cameraPositionInPositionSphereSpace - positionLevelData.getVertices()[i].pos);
			const auto faceIndex = mSphere->vectorToFaceIndex(localRotation, mRotationSphereLevel);
			const auto facePtr = rotationLevelData.getFaces()[faceIndex];

			// Interpolate the color using the face & barycentric coordinates
			const auto rawDataIndexA = (i * rotationLevelData.getNumberOfVertices()) + facePtr.vertices[0]->index;
			const auto rawDataIndexB = (i * rotationLevelData.getNumberOfVertices()) + facePtr.vertices[1]->index;
			const auto rawDataIndexC = (i * rotationLevelData.getNumberOfVertices()) + facePtr.vertices[2]->index;
			const auto rawDataA = mRawData->data()[rawDataIndexA];
			const auto rawDataB = mRawData->data()[rawDataIndexB];
			const auto rawDataC = mRawData->data()[rawDataIndexC];

			vec3 uvw(0.f);
			Barycentric(localRotation, facePtr.vertices[0]->pos, facePtr.vertices[1]->pos, facePtr.vertices[2]->pos, uvw);

			//result.push_back(rawDataA);
			//result.push_back(uvw);
			result.push_back(uvw.x * rawDataA + uvw.y * rawDataB + uvw.z * rawDataC);
		}

		return move(result);
	}
}
