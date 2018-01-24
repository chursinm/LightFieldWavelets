#pragma once
#include "ShaderManager.h"
#include "VRCamera.h"
#include "TrackballCamera.h"

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
	bool initializeGL();
	bool initializeSDL();
	bool initializeOpenVR();
	/**
		returns true, if the loop should cancel
	*/
	bool handleSDL();
	void render();

private: //functions
	void renderQuad(vr::Hmd_Eye eye);
	void renderStereoTargets();
	void renderCompanionWindow();
	void printerr();

private:
	bool m_bVblank=false;
	VRCamera m_Camera;
	TrackballCamera* m_pSecCamera;
	Uint64 m_LastFrameTime;
	bool m_ShiftDown;

private: // OpenGL
	FramebufferDesc m_LeftEyeFramebuffer;
	FramebufferDesc m_RightEyeFramebuffer;

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

