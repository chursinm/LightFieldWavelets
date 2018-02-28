#include "stdafx.h"
#include "VRCamera.h"
#include <glm/gtx/string_cast.hpp>

using namespace glm;

VRCamera::VRCamera(float nearClip, float farClip): m_NearClip(nearClip), m_FarClip(farClip)
{
}

VRCamera::~VRCamera()
{
}

void VRCamera::setup(vr::IVRSystem & vrInterface)
{
	m_LeftEyeProjection = getProjectionMatrix(vrInterface, vr::Eye_Left);
	m_RightEyeProjection = getProjectionMatrix(vrInterface, vr::Eye_Right);
	//WARN_ONCE(glm::to_string(m_RightEyeProjection))
	m_LeftEyePosition = getEyePositionMatrix(vrInterface, vr::Eye_Left);
	m_RightEyePosition = getEyePositionMatrix(vrInterface, vr::Eye_Right);
}

void VRCamera::update()
{
	vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
	vr::VRCompositor()->WaitGetPoses(trackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	if (trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
	{
		auto steamMatrix = trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking;
		m_HMDPosition = vec3(steamMatrix.m[0][3], steamMatrix.m[1][3], steamMatrix.m[2][3]);
		m_HMDTransformation = inverse(convertSteamMatrix(steamMatrix));
	}
	else
	{
		WARN_TIMED("No valid pose for the HMD", 2)
	}
}

vec3 VRCamera::getPosition(vr::Hmd_Eye eye)
{
	mat4x4& eyeMatrix = eye == vr::Eye_Left ? m_LeftEyePosition : m_RightEyePosition;
	auto eyeVector = inverse(m_HMDTransformation) * inverse(eyeMatrix) * vec4(0,0,0,1);
	return eyeVector.xyz * (1/eyeVector.w); 
}

mat4x4 VRCamera::getViewProjectionMatrix(vr::Hmd_Eye eye)
{
	switch (eye)
	{
	case vr::Eye_Left:
		return m_LeftEyeProjection * m_LeftEyePosition * m_HMDTransformation;
	case vr::Eye_Right:
		return m_RightEyeProjection * m_RightEyePosition * m_HMDTransformation;
	default:
		throw "invalid eye on camera.getmvp";
	}
}

glm::mat4x4 VRCamera::getViewMatrix(vr::Hmd_Eye eye)
{
	switch(eye)
	{
	case vr::Eye_Left:
		return m_LeftEyePosition * m_HMDTransformation;
	case vr::Eye_Right:
		return m_RightEyePosition * m_HMDTransformation;
	default:
		throw "invalid eye on camera.getmvp";
	}
}

glm::mat4x4 VRCamera::getProjectionMatrix(vr::Hmd_Eye eye)
{
	switch(eye)
	{
	case vr::Eye_Left:
		return m_LeftEyeProjection;
	case vr::Eye_Right:
		return m_RightEyeProjection;
	default:
		throw "invalid eye on camera.getmvp";
	}
}

mat4x4 VRCamera::getProjectionMatrix(vr::IVRSystem& vrInterface, vr::Hmd_Eye eye)
{
	vr::HmdMatrix44_t mat = vrInterface.GetProjectionMatrix(eye, m_NearClip, m_FarClip);

	return convertSteamMatrix(mat);
}

mat4x4 VRCamera::getEyePositionMatrix(vr::IVRSystem & vrInterface, vr::Hmd_Eye eye)
{
	vr::HmdMatrix34_t matEye = vrInterface.GetEyeToHeadTransform(eye);

	return inverse(convertSteamMatrix(matEye)); //TODO transpose
}

mat4x4 VRCamera::convertSteamMatrix(const vr::HmdMatrix34_t &mat)
{
	return mat4x4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], 0.0,
		mat.m[0][1], mat.m[1][1], mat.m[2][1], 0.0,
		mat.m[0][2], mat.m[1][2], mat.m[2][2], 0.0,
		mat.m[0][3], mat.m[1][3], mat.m[2][3], 1.0f
	);
}

mat4x4 VRCamera::convertSteamMatrix(const vr::HmdMatrix44_t & mat)
{
	return mat4x4(
		mat.m[0][0], mat.m[1][0], mat.m[2][0], mat.m[3][0],
		mat.m[0][1], mat.m[1][1], mat.m[2][1], mat.m[3][1],
		mat.m[0][2], mat.m[1][2], mat.m[2][2], mat.m[3][2],
		mat.m[0][3], mat.m[1][3], mat.m[2][3], mat.m[3][3]
	);
}
