#pragma once
#include "PlaneSampler.h"
class ColoredPlaneSampler :
	public PlaneSampler
{
public:
	ColoredPlaneSampler(const glm::vec3& hitColor, const glm::vec3& missColor, const Plane& plane);
	~ColoredPlaneSampler() = default;
	glm::vec3 sample(const Ray& ray) const override;

private:
	glm::vec3 mMissColor, mHitColor;
};
using ColoredPlaneSamplerPtr = std::unique_ptr<ColoredPlaneSampler>;

