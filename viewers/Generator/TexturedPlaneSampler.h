#pragma once
#include "PlaneSampler.h"

class TexturedPlaneSampler : public PlaneSampler
{
	using SurfacePtr = std::unique_ptr<SDL_Surface>;
public:


	TexturedPlaneSampler(const std::string& texturePath, const float uvScale, const glm::vec3& missColor, const Plane& plane);
	~TexturedPlaneSampler() = default;

	glm::vec3 sample(const Ray& ray) const override;
private:
	SurfacePtr mTexture;
	float mUvScale;
	glm::vec3 mMissColor;
};
using TexturedPlaneSamplerPtr = std::unique_ptr<TexturedPlaneSampler>;
