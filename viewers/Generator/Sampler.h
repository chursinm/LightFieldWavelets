#pragma once
#include "Ray.h"
namespace Generator
{
	namespace Sampler
	{
		class Sampler
		{
		public:
			virtual glm::vec4 sample(const Ray& ray) const = 0;
		};
	}
}