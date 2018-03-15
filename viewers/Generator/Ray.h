#pragma once
namespace Generator
{
	struct Ray
	{
		Ray(const glm::vec3& origin, const glm::vec3& direction) : mOrigin(origin), mDirection(glm::normalize(direction)) {}
		Ray(const glm::vec3& origin, const glm::vec3& direction, void* noNormalize) : mOrigin(origin), mDirection(direction){}
		glm::vec3 mOrigin, mDirection;
		glm::vec3 atDistance(const float distance) const
		{
			return mOrigin + mDirection * distance;
		}
	};
}