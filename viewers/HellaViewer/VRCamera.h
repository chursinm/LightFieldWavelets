#pragma once

#include <openvr.h>

class VRCamera
{
public:
	VRCamera(float nearClip=.1f, float farClip=30.f);
	~VRCamera();
	void setup(vr::IVRSystem& vrInterface);
	void update();
	glm::vec3 getPosition(vr::Hmd_Eye eye);
	glm::mat4x4 getMVP(vr::Hmd_Eye eye);
private:
	float	m_NearClip, m_FarClip;
	glm::mat4x4	m_LeftEyeProjection, m_RightEyeProjection,
			m_LeftEyePosition, m_RightEyePosition,
			m_HMDTransformation;
	glm::vec3 m_HMDPosition;
	glm::mat4x4	getProjectionMatrix(vr::IVRSystem& vrInterface, vr::Hmd_Eye eye);
	glm::mat4x4	getEyePositionMatrix(vr::IVRSystem& vrInterface, vr::Hmd_Eye eye);
	glm::mat4x4	convertSteamMatrix(const vr::HmdMatrix34_t &matPose);
	glm::mat4x4	convertSteamMatrix(const vr::HmdMatrix44_t &matPose);
};
