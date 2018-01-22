#pragma once

#include <openvr.h>
using namespace glm;

class VRCamera
{
public:
	VRCamera(float nearClip=.1f, float farClip=30.f);
	~VRCamera();
	void setup(vr::IVRSystem& vrInterface);
	void update();
	mat4x4 getMVP(vr::Hmd_Eye eye);
private:
	float	m_NearClip, m_FarClip;
	mat4x4	m_LeftEyeProjection, m_RightEyeProjection,
			m_LeftEyePosition, m_RightEyePosition,
			m_HMDPosition;
	mat4x4	getProjectionMatrix(vr::IVRSystem& vrInterface, vr::Hmd_Eye eye);
	mat4x4	getEyePositionMatrix(vr::IVRSystem& vrInterface, vr::Hmd_Eye eye);
	mat4x4	convertSteamMatrix(const vr::HmdMatrix34_t &matPose);
	mat4x4	convertSteamMatrix(const vr::HmdMatrix44_t &matPose);
};

