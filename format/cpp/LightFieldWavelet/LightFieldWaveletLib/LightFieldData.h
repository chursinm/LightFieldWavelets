#pragma once
#include "SubdivisionSphere.h"
#include "LevelMatrix.h"
#include <memory>


using namespace glm;


namespace LightField
{
	
	class LightFieldData
	{
	public:
		LightFieldData(const std::shared_ptr <SubdivisionSphere> subsphere);
		std::shared_ptr<LevelMatrix> getLevelMatrix(int level) const { return levelMatices[level]; } 
		static void Barycentric(const vec3& p, const vec3& a, const vec3& b, const vec3& c, vec3& uvw)
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

	private:
		std::shared_ptr <SubdivisionSphere> subSphere;
		std::vector <std::shared_ptr<LevelMatrix>> levelMatices;
		
	};
}