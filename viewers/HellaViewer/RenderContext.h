#pragma once
#include "ShaderManager.h"
#include "VRCamera.h"
#include "TrackballCamera.h"
#include "Renderer.h"
#include "Signal.h"

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
	explicit RenderContext(bool vsync);
	~RenderContext();
	void attachRenderer(Renderer& rend);
	bool initialize();
	/**
		returns true, if the loop should cancel
	*/
	bool handleSDL();
	void render();

	Signal<> postInitialize;
	Signal<const glm::mat4x4&, const glm::vec3&> onRenderEyeTexture;
	Signal<double> onFrameStart;
	Signal<SDL_Keymod,SDL_Keycode> onKeyPress;

private: //functions
	bool initializeGL();
	bool initializeSDL();
	bool initializeOpenVR();
	void renderQuad(vr::Hmd_Eye eye);
	void renderStereoTargets();
	void renderCompanionWindow();
	void printerr(const char * file, const int line);

private:
	bool m_Initialized;
	bool m_RenderRightEye;
	bool m_bVblank;
	// TODO common Camera Interface
	VRCamera m_VRCamera;
	TrackballCamera* m_pTrackballCamera;
	bool m_VREnabled;
	Uint64 m_LastFrameTime, m_FrameCounter;
	double m_AccumulatedFrameTime;

private: // OpenGL
	FramebufferDesc m_LeftEyeFramebuffer;
	FramebufferDesc m_RightEyeFramebuffer;

	GLuint m_DesktopProgram;

	bool createFrameBuffer(int nWidth, int nHeight, FramebufferDesc &framebufferDesc);
	void printGLInteger(const std::string & name, GLint constant);

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