#pragma once
#include "ShaderManager.h"
#include "VRCamera.h"

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
	/**
		returns true, if the loop should cancel
	*/
	bool handleSDL();
	void render();


private:
	bool m_bVblank=false;
	void printerr();

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
};

