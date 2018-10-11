#pragma once
namespace Generator
{
	struct Ray
	{
		Ray(const glm::vec3& origin, const glm::vec3& direction) : mOrigin(origin), mDirection(glm::normalize(direction)) {}
		Ray(const glm::vec3& origin,
			const glm::vec3& direction,
			const double intensityIn,
			const double CIExIn,
			const double CIEyIn,
			const unsigned long layerIn) : 
				mOrigin(origin), 
				mDirection(glm::normalize(direction)),	
				intensity(intensityIn),
				CIEx(CIExIn),
				CIEy(CIEyIn),
				layer(layerIn) {}
		Ray(const glm::vec3& origin, const glm::vec3& direction, const double intesityIn) : mOrigin(origin), mDirection(glm::normalize(direction)), intensity(intesityIn){}
		//Ray(const glm::vec3& origin, const glm::vec3& direction, void* noNormalize) : mOrigin(origin), mDirection(direction){}
		glm::vec3 mOrigin, mDirection;
		double intensity;     /// Intensity of the ray in relative lumen units
		double CIEx;          /// CIE1931 x coordinate of the rays color
		double CIEy;          /// CIE1931 y coordinate of the rays color
		double wavelength;    /// rays wavelength 
		unsigned long layer;  /// index of the subset of rays to which the ray belongs
		static double adj( double C) {
			if (abs(C) < 0.0031308) {
				return 12.92 * C;
			}
			return 1.055 * pow(C, 0.41666) - 0.055;
		}
		glm::vec3 getRGB()
		{	
			double X = (CIEx / CIEy)*intensity;
			double Y = intensity;
			double Z = (1 - CIEx - CIEy)*intensity / CIEy;
			double R = 3.2404542*	X - 1.5371385*Y - 0.4985314*Z;
			double G = -0.9692660*	X + 1.8760108*Y + 0.0415560*Z;
			double B = 0.0556434*	X - 0.2040259*Y + 1.0572252*Z;
			return glm::vec3(adj(std::max(R,0.0)), adj(std::max(G, 0.0)), adj(std::max(B, 0.0)));
		}
		glm::vec3 atDistance(const float distance) const
		{
			return mOrigin + mDirection * distance;
		}
	};
}