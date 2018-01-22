#include "stdafx.h"
#include "VRCamera.h"


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
	m_LeftEyePosition = getEyePositionMatrix(vrInterface, vr::Eye_Left);
	m_RightEyePosition = getEyePositionMatrix(vrInterface, vr::Eye_Right);
}

void VRCamera::update()
{
	vr::TrackedDevicePose_t trackedDevicePose[vr::k_unMaxTrackedDeviceCount];
	vr::VRCompositor()->WaitGetPoses(trackedDevicePose, vr::k_unMaxTrackedDeviceCount, NULL, 0);

	if (trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].bPoseIsValid)
	{
		m_HMDPosition = inverse(convertSteamMatrix(trackedDevicePose[vr::k_unTrackedDeviceIndex_Hmd].mDeviceToAbsoluteTracking));
	}
	else
	{
		std::cout << "No valid pose for the HMD" << std::endl;
	}
}

mat4x4 VRCamera::getMVP(vr::Hmd_Eye eye)
{
	switch (eye)
	{
	case vr::Eye_Left:
		return m_LeftEyeProjection * m_LeftEyePosition * m_HMDPosition;
	case vr::Eye_Right:
		return m_RightEyeProjection * m_RightEyePosition * m_HMDPosition;
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

	return inverse(convertSteamMatrix(matEye));
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
