#include "stdafx.h"
#include "RenderContext.h"


RenderContext::RenderContext()
{
}



RenderContext::~RenderContext()
{
	//TODO clear GL resources

	std::cout << "shutting down" << std::endl;

	if (m_pHMD)
	{
		vr::VR_Shutdown();
		m_pHMD = NULL;
	}

	SDL_StopTextInput();
	if (m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	SDL_Quit();
}

bool RenderContext::initializeGL()
{
	std::cout << "GL Version: " << glGetString(GL_VERSION) << std::endl;

	// Some default GL settings.
	// TODO adapt to our program
	glClearColor(0, 0, 0, 255);
	glClearDepth(1.0f);
	/*glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);
	glCullFace(GL_BACK);
	glDisable(GL_CULL_FACE);
	glShadeModel(GL_SMOOTH);
	glEnable(GL_MULTISAMPLE);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);
	glEnable(GL_NORMALIZE);*/

	// Create GL programs
	ShaderManager& psm = ShaderManager::instance();
	m_StereoProgram = psm.from("shader/stereo.vert", "shader/stereo.frag");
	m_DesktopProgram = psm.from("shader/desktop.vert", "shader/desktop.frag");
	if (m_DesktopProgram == 0 || m_StereoProgram == 0)
		return false;

	// Create GL buffers
	if (!createFrameBuffer(m_RenderWidth, m_RenderHeight, m_LeftEyeFramebuffer) ||
		!createFrameBuffer(m_RenderWidth, m_RenderHeight, m_RightEyeFramebuffer))
		return false;

	return true;
}

bool RenderContext::initializeSDL()
{
	// initialize the sdl system and create a window
	if (SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	//SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);


	Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
	int nWindowPosX = 700;
	int nWindowPosY = 100;
	m_pCompanionWindow = SDL_CreateWindow("Hella Viewer", nWindowPosX, nWindowPosY, m_nCompanionWindowWidth, m_nCompanionWindowHeight, unWindowFlags);
	if (m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if (m_pContext == NULL)
	{
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}


	glewExperimental = GL_TRUE; // otherwise glGenFramebuffer is not defined
	GLenum err = glewInit();
	if (GLEW_OK != err)
	{
		fprintf(stderr, "Error initializing glew: %s\n", glewGetErrorString(err));
		return false;
	}
	printerr();

	if (SDL_GL_SetSwapInterval(m_bVblank ? 1 : 0) < 0)
	{
		printf("%s - Warning: Unable to set VSync! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}


	SDL_SetWindowTitle(m_pCompanionWindow, "Hello VR");

	SDL_StartTextInput();
	SDL_ShowCursor(SDL_DISABLE);

	return true;
}

bool RenderContext::initializeOpenVR()
{
	// Loading the SteamVR Runtime
	vr::EVRInitError eError = vr::VRInitError_None;
	m_pHMD = vr::VR_Init(&eError, vr::VRApplication_Scene);

	if (eError != vr::VRInitError_None || m_pHMD == nullptr)
	{
		m_pHMD = NULL;
		char buf[1024];
		sprintf_s(buf, sizeof(buf), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, NULL);
		return false;
	}

	vr::EVRInitError peError = vr::VRInitError_None;

	if (!vr::VRCompositor())
	{
		printf("Compositor initialization failed. See log file for details\n");
		return false;
	}

	m_pHMD->GetRecommendedRenderTargetSize(&m_RenderWidth, &m_RenderHeight);

	m_Camera.setup(*m_pHMD);

	return true;
}

bool RenderContext::handleSDL()
{
	SDL_Event sdlEvent;
	bool quitProgram = false;

	while (SDL_PollEvent(&sdlEvent) != 0)
	{
		if (sdlEvent.type == SDL_QUIT)
		{
			quitProgram = true;
		}
		else if (sdlEvent.type == SDL_KEYDOWN)
		{
			if (sdlEvent.key.keysym.sym == SDLK_ESCAPE
				|| sdlEvent.key.keysym.sym == SDLK_q)
			{
				quitProgram = true;
			}
			else if (sdlEvent.key.keysym.sym == SDLK_SPACE)
			{
				auto leftpos = m_Camera.getPosition(vr::Eye_Left);
				auto rightpos = m_Camera.getPosition(vr::Eye_Right);

				std::cout << "Left eye position: " << leftpos.x << ", " << leftpos.y << ", " << leftpos.z << std::endl;
				std::cout << "Right eye position: " << rightpos.x << ", " << rightpos.y << ", " << rightpos.z << std::endl << std::endl;
			}
		}
		else if (sdlEvent.type == SDL_WINDOWEVENT &&
			sdlEvent.window.event == SDL_WINDOWEVENT_RESIZED)
		{
			int width = sdlEvent.window.data1, 
				height = sdlEvent.window.data2;
			resize(width, height);
		}
	}

	return quitProgram;
}

void RenderContext::render()
{
	// Render
	{
		renderStereoTargets();
		renderCompanionWindow();

		vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)m_LeftEyeFramebuffer.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
		vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)m_RightEyeFramebuffer.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
		vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
	}

	if (m_bVblank)
	{
		//$ HACKHACK. From gpuview profiling, it looks like there is a bug where two renders and a present
		// happen right before and after the vsync causing all kinds of jittering issues. This glFinish()
		// appears to clear that up. Temporary fix while I try to get nvidia to investigate this problem.
		// 1/29/2014 mikesart
		glFinish();
	}

	// SwapWindow
	{
		SDL_GL_SwapWindow(m_pCompanionWindow);
	}

	// Clear
	{
		// We want to make sure the glFinish waits for the entire present to complete, not just the submission
		// of the command. So, we do a clear here right here so the glFinish will wait fully for the swap.
		glClearColor(0, 0, 0, 1);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	}

	// Flush and wait for swap.
	if (m_bVblank)
	{
		glFlush();
		glFinish();
	}


	printerr();
	m_Camera.update();
}

void RenderContext::renderQuad(vr::Hmd_Eye eye)
{

	//glDisable(GL_DEPTH_TEST);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glUseProgram(m_StereoProgram);
	glUniform1ui(glGetUniformLocation(m_StereoProgram, "eye"), eye);
	auto mvp = m_Camera.getMVP(eye);
	glUniformMatrix4fv(glGetUniformLocation(m_StereoProgram, "mvp"), 1, GL_FALSE, &mvp[0][0]);

	glBegin(GL_QUADS);
	glVertex3f(-10.0f, 0.f, -10.0f);
	glVertex3f(10.0f, 0.f, -10.0f);
	glVertex3f(10.0f, 0.f, 10.0f);
	glVertex3f(-10.0f, 0.f, 10.0f);
	glEnd();

	/*glBegin(GL_QUADS);
	glVertex2f(-1.0f, -1.0f);
	glVertex2f(1.0f, -1.0f);
	glVertex2f(1.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();*/

	glUseProgram(0);
}

void RenderContext::renderStereoTargets()
{
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
	glEnable(GL_MULTISAMPLE);

	// Left Eye
	glBindFramebuffer(GL_FRAMEBUFFER, m_LeftEyeFramebuffer.m_nRenderFramebufferId);
	glViewport(0, 0, m_RenderWidth, m_RenderHeight);
	renderQuad(vr::Eye_Left);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_MULTISAMPLE);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_LeftEyeFramebuffer.m_nRenderFramebufferId);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_LeftEyeFramebuffer.m_nResolveFramebufferId);

	glBlitFramebuffer(0, 0, m_RenderWidth, m_RenderHeight, 0, 0, m_RenderWidth, m_RenderHeight,
		GL_COLOR_BUFFER_BIT,
		GL_LINEAR);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);

	glEnable(GL_MULTISAMPLE);

	// Right Eye
	glBindFramebuffer(GL_FRAMEBUFFER, m_RightEyeFramebuffer.m_nRenderFramebufferId);
	glViewport(0, 0, m_RenderWidth, m_RenderHeight);
	renderQuad(vr::Eye_Right);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	glDisable(GL_MULTISAMPLE);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, m_RightEyeFramebuffer.m_nRenderFramebufferId);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, m_RightEyeFramebuffer.m_nResolveFramebufferId);

	glBlitFramebuffer(0, 0, m_RenderWidth, m_RenderHeight, 0, 0, m_RenderWidth, m_RenderHeight,
		GL_COLOR_BUFFER_BIT,
		GL_LINEAR);

	glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

void RenderContext::renderCompanionWindow()
{
	glDisable(GL_DEPTH_TEST);
	glViewport(0, 0, m_nCompanionWindowWidth, m_nCompanionWindowHeight);

	glUseProgram(m_DesktopProgram);

	// render left eye
	glBindTexture(GL_TEXTURE_2D, m_LeftEyeFramebuffer.m_nResolveTextureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(-1.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(0.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(-1.0f, 1.0f);
	glEnd();

	// render right eye 
	glBindTexture(GL_TEXTURE_2D, m_RightEyeFramebuffer.m_nResolveTextureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glBegin(GL_QUADS);
	glTexCoord2f(0.0f, 0.0f);
	glVertex2f(0.0f, -1.0f);
	glTexCoord2f(1.0f, 0.0f);
	glVertex2f(1.0f, -1.0f);
	glTexCoord2f(1.0f, 1.0f);
	glVertex2f(1.0f, 1.0f);
	glTexCoord2f(0.0f, 1.0f);
	glVertex2f(0.0f, 1.0f);
	glEnd();

	glUseProgram(0);
}

void RenderContext::printerr()
{
	GLenum err = glGetError();
	if (err != GL_NO_ERROR) 
	printf("Error: %i - %s\n", err, gluErrorString(err));
}

void RenderContext::resize(int width, int height)
{
	m_nCompanionWindowWidth = width;
	m_nCompanionWindowHeight = height;
	std::cout << "resizing window to " << width << "x" << height << std::endl;

}

bool RenderContext::createFrameBuffer(int nWidth, int nHeight, FramebufferDesc & framebufferDesc)
{
	glGenFramebuffers(1, &framebufferDesc.m_nRenderFramebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nRenderFramebufferId);

	glGenRenderbuffers(1, &framebufferDesc.m_nDepthBufferId);
	glBindRenderbuffer(GL_RENDERBUFFER, framebufferDesc.m_nDepthBufferId);
	glRenderbufferStorageMultisample(GL_RENDERBUFFER, 4, GL_DEPTH_COMPONENT, nWidth, nHeight);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, framebufferDesc.m_nDepthBufferId);

	glGenTextures(1, &framebufferDesc.m_nRenderTextureId);
	glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId);
	glTexImage2DMultisample(GL_TEXTURE_2D_MULTISAMPLE, 4, GL_RGBA8, nWidth, nHeight, true);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D_MULTISAMPLE, framebufferDesc.m_nRenderTextureId, 0);

	glGenFramebuffers(1, &framebufferDesc.m_nResolveFramebufferId);
	glBindFramebuffer(GL_FRAMEBUFFER, framebufferDesc.m_nResolveFramebufferId);

	glGenTextures(1, &framebufferDesc.m_nResolveTextureId);
	glBindTexture(GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, nWidth, nHeight, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, framebufferDesc.m_nResolveTextureId, 0);

	// check FBO status
	GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
	if (status != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer not complete" << std::endl;
		return false;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}
