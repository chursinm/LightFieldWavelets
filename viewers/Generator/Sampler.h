#pragma once

namespace Generator
{
	namespace Sampler
	{
		class Sampler
		{
		public:
			struct Ray
			{
				Ray(const glm::vec3& origin, const glm::vec3& direction) : mOrigin(origin), mDirection(glm::normalize(direction)) {}
				glm::vec3 mOrigin, mDirection;
				glm::vec3 atDistance(const float distance) const
				{
					return mOrigin + mDirection * distance;
				}
			};
			virtual glm::vec4 sample(const Ray& ray) const = 0;
		};
	}
}