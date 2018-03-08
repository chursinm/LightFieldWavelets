#pragma once
class Sampler
{
public:
	struct Ray
	{
		Ray(const glm::vec3& origin, const glm::vec3& direction) : mOrigin(origin), mDirection(glm::normalize(direction)) {}
		glm::vec3 mOrigin, mDirection;
	};
	virtual glm::vec3 sample(const Ray& ray) const = 0;
};
using SamplerPtr = std::unique_ptr<Sampler>;
