#pragma once
#include "ShaderManager.h"
#include "VRCamera.h"
#include "TrackballCamera.h"
#include "CameraArrayRenderer.h"

struct FramebufferDesc
{
	GLuint m_nDepthBufferId;
	GLuint m_nRenderTextureId;
	GLuint m_nRenderFramebufferId;
	GLuint m_nResolveTextureId;
	GLuint m_nResolveFramebufferId;
};

class RenderContext
{
	
public:
	RenderContext();
	~RenderContext();
	bool initialize();
	/**
		returns true, if the loop should cancel
	*/
	bool handleSDL();
	void render();

private: //functions
	bool initializeGL();
	bool initializeSDL();
	bool initializeOpenVR();
	void renderQuad(vr::Hmd_Eye eye);
	void renderStereoTargets();
	void renderCompanionWindow();
	void printerr(const char * file, const int line);


private: // external rendering components
	CameraArrayRenderer m_CameraArrayRenderer;

private:
	bool m_RenderRightEye;
	bool m_bVblank=false;
	// TODO common Camera Interface
	VRCamera m_VRCamera;
	TrackballCamera* m_pTrackballCamera;
	bool m_VREnabled;
	Uint64 m_LastFrameTime, m_FrameCounter;
	double m_AccumulatedFrameTime;
	bool m_ShiftDown;

private: // OpenGL
	FramebufferDesc m_LeftEyeFramebuffer;
	FramebufferDesc m_RightEyeFramebuffer;

	GLuint m_BlitTriangleVB, m_BlitTriangleVAO;
	GLuint m_StereoProgram;
	GLuint m_DesktopProgram;

	bool createFrameBuffer(int nWidth, int nHeight, FramebufferDesc &framebufferDesc);

private: // SDL
	SDL_Window * m_pCompanionWindow;
	uint32_t m_nCompanionWindowWidth=1280;
	uint32_t m_nCompanionWindowHeight=720;
	SDL_GLContext m_pContext;
	void resize(int width, int height);

private: // OpenVR
	vr::IVRSystem* m_pHMD;
	uint32_t m_RenderWidth, m_RenderHeight;
};