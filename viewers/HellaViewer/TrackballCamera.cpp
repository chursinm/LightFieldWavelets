#include "stdafx.h"
#include "TrackballCamera.h"

using namespace glm;


TrackballCamera::TrackballCamera(glm::uvec2 viewportSize) : m_Position(0.0f, 1.0f, 0.0f), m_Rotation(1.f, 0.f, 0.f, 0.f), m_ViewportSize(viewportSize)
{
}

TrackballCamera::TrackballCamera(unsigned int viewportWidth, unsigned int viewportHeight) : m_Position(0.0f, 1.0f, 0.0f), m_Rotation(), m_ViewportSize(uvec2(viewportWidth, viewportHeight))
{
}

TrackballCamera::~TrackballCamera()
{
}

void TrackballCamera::pan(float dx, float dy)
{
	vec3 eye = m_Rotation * c_Forward;
	vec3 aDir = normalize(-eye);
	vec3 rup = normalize(m_Rotation * c_Up);
	vec3 aRight = normalize(cross(aDir, rup));
	vec3 aUp = cross(aDir, aRight);
	m_Position += aRight * dx + aUp * dy;
}

void TrackballCamera::move(float dx, float dz)
{
	vec3 eye = m_Rotation * c_Forward;
	vec3 aDir = normalize(-eye);
	vec3 rside = normalize(m_Rotation * c_Side);
	vec3 up = normalize(cross(aDir, rside));
	vec3 right = cross(aDir, up);
	m_Position += eye * dx + right * dz;
}

void TrackballCamera::zoom(float dx)
{
	move(dx, 0.f);
}

void TrackballCamera::rotate(const uvec2& currentCursorPosition, const uvec2& lastCursorPosition)
{
	static const auto epsilon = 1e-6f;
	vec3 current = calculateTrackballPosition(currentCursorPosition);
	vec3 last = calculateTrackballPosition(lastCursorPosition);

	// see: http://web.cse.ohio-state.edu/~hwshen/781/Site/Slides_files/trackball.pdf
	auto dotP = dot(current, last);
	if (abs(dotP - 1.f) < epsilon) return; // no rotation happened
	float rotationAngle = std::acos(dotP);
	vec3 rotationAxis = normalize(cross(last, current));
	
	m_Rotation = glm::rotate(m_Rotation, rotationAngle, rotationAxis);
	m_Rotation = normalize(m_Rotation);
}


vec3 TrackballCamera::calculateTrackballPosition(const uvec2& point) const
{
	// Get screen coordinates in [0,1] range
	auto screencoord = vec2(static_cast<float>(point.x) / static_cast<float>(m_ViewportSize.x), static_cast<float>(point.y) / static_cast<float>(m_ViewportSize.y));
	// Transform them to [-1,1] range
	screencoord = screencoord * 2.f - vec2(1.f, 1.f);

	// Transform to 3D unit sphere coordinates
	auto spherecoord = vec3(screencoord, 0.f);
	float zSqrt = 1.0f - dot(spherecoord, spherecoord);
	if (zSqrt <= 0.0f) // out of sphere, clip to outer circle
	{
		spherecoord = normalize(spherecoord);
	}
	else // |r| = 1 => z = sqrt(1-x^2-y^2)
	{
		spherecoord.z = std::sqrt(zSqrt);
	}
	return spherecoord;
}

mat4x4 TrackballCamera::viewMatrix() const
{
	vec3 forward = normalize(m_Rotation * c_Forward);
	vec3 up = normalize(m_Rotation * c_Up);
	vec3 right = normalize(m_Rotation * c_Side);

	auto positionMatrix = mat4x4(1.f);
	positionMatrix[3] = vec4(-m_Position, 1.f); // setzt letzte Spalte
	
	auto rotationMatrix = mat4x4(1.f); 
	rotationMatrix[0][0] = right.x;
	rotationMatrix[0][1] = up.x;
	rotationMatrix[0][2] = forward.x;

	rotationMatrix[1][0] = right.y;
	rotationMatrix[1][1] = up.y;
	rotationMatrix[1][2] = forward.y;

	rotationMatrix[2][0] = right.z;
	rotationMatrix[2][1] = up.z;
	rotationMatrix[2][2] = forward.z;

	return rotationMatrix * positionMatrix;
}

mat4x4 TrackballCamera::projectionMatrix() const
{
	// taken from http://nehe.gamedev.net/article/replacement_for_gluperspective/21002/
	double zFar = 25.;
	double zNear = 0.0045;
	double fovy = 106.;
	const double pi = 3.1415926535897932384626433832795;
	double fW, fH;
	fH = tan(fovy / 360. * pi) * zNear;
	fW = fH * ((double)m_ViewportSize.x / (double)m_ViewportSize.y);

	// taken from http://www.manpagez.com/man/3/glFrustum/
	double left = -fW;
	double right = fW;
	double bottom = -fH;
	double top = fH;
	//zNear = zNear; zFar = zfar

	double A = (right + left) / (right - left);
	double B = (top + bottom) / (top - bottom);
	double C = -(zFar + zNear) / (zFar - zNear);
	double D = -(2 * zFar * zNear) / (zFar - zNear);
	double E = (2 * zNear) / (right - left);
	double F = (2 * zNear) / (top - bottom);

	vec4 c1(E, 0., 0., 0.);
	vec4 c2(0., F, 0., 0.);
	vec4 c3(A, B, C, -1.);
	vec4 c4(0., 0., D, 0.);

	return mat4x4(c1,c2,c3,c4);
}

void TrackballCamera::reset()
{
	m_Rotation = quat(1.f, 0.f, 0.f, 0.f);
	m_Position = vec3(0.0f, 1.0f, 0.0f);
}

vec3 TrackballCamera::getPosition() const
{
	return m_Position;
}
/*
FrustumCorners TrackballCamera::frustum(double far) const
{
	vec3 eye = rotation.rotatedVector(forward);
	vec3 rup = rotation.rotatedVector(up);
	rup.normalize();
	eye.normalize();
	vec3 rright = vec3::crossProduct(eye, rup);
	rright.normalize();

	double near = 0.045;
	double fovy = 45.;

	double hheight = tan(fovy*0.5); //half the height of the frustum at z=1;
	double hwidth = hheight * m_aspect; //same for width
	double hFarHeight = far * hheight;
	double hFarWidth = far * hwidth;
	double hNearHeight = near * hheight;
	double hNearWidth = near * hwidth;

	FrustumCorners ret;
	vec3 fc = m_position + eye * far;
	ret.ftl = fc + (rup * hFarHeight) - (rright * hFarWidth);
	ret.ftr = fc + (rup * hFarHeight) + (rright * hFarWidth);
	ret.fbl = fc - (rup * hFarHeight) - (rright * hFarWidth);
	ret.fbr = fc - (rup * hFarHeight) + (rright * hFarWidth);
	vec3 nc = m_position + eye * near;
	ret.ntl = nc + (rup * hNearHeight) - (rright * hNearWidth);
	ret.ntr = nc + (rup * hNearHeight) + (rright * hNearWidth);
	ret.nbl = nc - (rup * hNearHeight) - (rright * hNearWidth);
	ret.nbr = nc - (rup * hNearHeight) + (rright * hNearWidth);

	return ret;
}
*/
