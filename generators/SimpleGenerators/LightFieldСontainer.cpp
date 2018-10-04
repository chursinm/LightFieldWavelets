#include "stdafx.h"
#include "LightFieldСontainer.h"


using namespace glm;

Generator::LightFieldСontainer::LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, std::shared_ptr<Sampler::Sampler> samplerIn):
	subdivistionSphere(sphereIn),
	sampler(samplerIn),
	lightFieldData(subdivistionSphere)
{
	
}

Generator::LightFieldСontainer::LightFieldСontainer(std::shared_ptr<LightField::SubdivisionSphere> sphereIn, std::shared_ptr<Generator::RWRReader> rwrReaderIn):
	subdivistionSphere(sphereIn),
	rwrReader(rwrReaderIn),
	lightFieldData(subdivistionSphere)
{

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
