#pragma once
namespace Generator
{
	struct Ray
	{
		Ray(const glm::vec3& origin, const glm::vec3& direction) : mOrigin(origin), mDirection(glm::normalize(direction)) {}
		Ray(const glm::vec3& origin, const glm::vec3& direction, const double intesityIn) : mOrigin(origin), mDirection(glm::normalize(direction)), intensity(intesityIn){}
		//Ray(const glm::vec3& origin, const glm::vec3& direction, void* noNormalize) : mOrigin(origin), mDirection(direction){}
		glm::vec3 mOrigin, mDirection;
		double intensity;     /// Intensity of the ray in relative lumen units
		double CIEx;          /// CIE1931 x coordinate of the rays color
		double CIEy;          /// CIE1931 y coordinate of the rays color
		double wavelength;    /// rays wavelength 
		unsigned long layer;  /// index of the subset of rays to which the ray belongs
		glm::vec3 atDistance(const float distance) const
		{
			return mOrigin + mDirection * distance;
		}
	};
}