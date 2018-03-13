#pragma once
#include "PlaneSampler.h"

namespace Generator
{
	namespace Sampler
	{
		class CheckerboardSampler : public PlaneSampler
		{
			using SurfacePtr = std::unique_ptr<SDL_Surface>;
		public:
			CheckerboardSampler(float checkerSquareLength, const glm::vec3& missColor, const Plane& plane);
			~CheckerboardSampler() = default;

			glm::vec4 sample(const Ray& ray) const override;
		private:
			float mCheckerSquareLength;
			glm::vec3 mMissColor;
		};
	}
}