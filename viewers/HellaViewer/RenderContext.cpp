#include "stdafx.h"
#include "RenderContext.h"
#include "Blit.h"

#define PRINT_GL_INTEGER( name ) { printGLInteger(#name, name); }

RenderContext::RenderContext(bool vsync) : m_Initialized(false), m_RenderRightEye(true), m_bVblank(vsync), m_LastFrameTime(0u), m_AccumulatedFrameTime(0.0), m_FrameCounter(0), m_pTrackballCamera(nullptr), m_VREnabled(false)
{
}



RenderContext::~RenderContext()
{
	//TODO clear GL resources

	std::cout << "shutting down" << std::endl;

	if(m_pHMD)
	{
		vr::VR_Shutdown();
		m_pHMD = NULL;
	}

	SDL_StopTextInput();
	if(m_pCompanionWindow)
	{
		SDL_DestroyWindow(m_pCompanionWindow);
		m_pCompanionWindow = NULL;
	}

	SDL_Quit();
}

void RenderContext::attachRenderer(Renderer & rend)
{
	onRenderEyeTexture([&rend](auto vp, auto eyePosWorld) { rend.render(vp, eyePosWorld); });
	onFrameStart([&rend](auto timeStep) { rend.update(timeStep); });
	if(m_Initialized)
	{
		rend.initialize();
	}
	else
	{
		postInitialize([&rend] { rend.initialize(); });
	}
}

bool RenderContext::initialize()
{
	bool success = true;
	if(!initializeSDL()) { success = false; }
	if(initializeOpenVR())
	{
		m_VREnabled = true;
	}
	else
	{
		m_VREnabled = false;
		m_RenderWidth = 1280u;
		m_RenderHeight = 720u;
		m_pTrackballCamera = new TrackballCamera(m_RenderWidth, m_RenderHeight);
	}
	if(!initializeGL()) { success = false; }

	postInitialize();
	m_Initialized = success;

	return success;
}

bool RenderContext::initializeGL()
{
	std::cout << "GL Version: " << glGetString(GL_VERSION) << std::endl;

	// Some default GL settings.
	glClearColor(0, 0, 0, 255);
	glClearDepth(1.0f);

	// Create GL programs
	ShaderManager& psm = ShaderManager::instance();
	m_DesktopProgram = psm.from("shader/desktop.vert", "shader/desktop.frag");
	if(m_DesktopProgram == 0)
		return false;

	// Create GL buffers
	if(!createFrameBuffer(m_RenderWidth, m_RenderHeight, m_LeftEyeFramebuffer) ||
		!createFrameBuffer(m_RenderWidth, m_RenderHeight, m_RightEyeFramebuffer))
		return false;

	return true;
}

bool RenderContext::initializeSDL()
{
	// initialize the sdl system and create a window
	if(SDL_Init(SDL_INIT_VIDEO | SDL_INIT_TIMER) < 0)
	{
		printf("%s - SDL could not initialize! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
	//SDL_GL_SetAttribute( SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY );
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLEBUFFERS, 0);
	SDL_GL_SetAttribute(SDL_GL_MULTISAMPLESAMPLES, 0);

	SDL_LogSetAllPriority(SDL_LOG_PRIORITY_INFO);
	SDL_LogSetOutputFunction([](void *userdata, int category, SDL_LogPriority priority, const char *message)
	{
		std::cout << "SDL message: " << message << std::endl;
	}, 0);


	Uint32 unWindowFlags = SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN | SDL_WINDOW_RESIZABLE;
	auto monitorIndex = SDL_GetNumVideoDisplays() - 1;
	m_pCompanionWindow = SDL_CreateWindow("Hella Viewer", SDL_WINDOWPOS_UNDEFINED_DISPLAY(monitorIndex), SDL_WINDOWPOS_UNDEFINED_DISPLAY(monitorIndex), m_nCompanionWindowWidth, m_nCompanionWindowHeight, unWindowFlags);
	if(m_pCompanionWindow == NULL)
	{
		printf("%s - Window could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}

	m_pContext = SDL_GL_CreateContext(m_pCompanionWindow);
	if(m_pContext == NULL)
	{
		printf("%s - OpenGL context could not be created! SDL Error: %s\n", __FUNCTION__, SDL_GetError());
		return false;
	}


	glewExperimental = GL_TRUE; // otherwise glGenFramebuffer is not defined
	GLenum err = glewInit();
	GLenum glerr = glGetError();
	if(GLEW_OK != err && GL_INVALID_ENUM != glerr) // glewExperimental throws invalid enum, see https://www.khronos.org/opengl/wiki/OpenGL_Loading_Library
	{
		fprintf(stderr, "Error initializing glew: %s\n gl: %s", glewGetErrorString(err), gluErrorString(glerr));
		return false;
	}

	if(SDL_GL_SetSwapInterval(m_bVblank ? 1 : 0) < 0)
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

	if(eError != vr::VRInitError_None || m_pHMD == nullptr)
	{
		m_pHMD = NULL;
		//char buf[1024];
		//sprintf_s(buf, sizeof(buf), "Unable to init VR runtime: %s", vr::VR_GetVRInitErrorAsEnglishDescription(eError));
		//SDL_ShowSimpleMessageBox(SDL_MESSAGEBOX_ERROR, "VR_Init Failed", buf, NULL);
		std::cout << "Unable to init VR runtime: " << vr::VR_GetVRInitErrorAsEnglishDescription(eError) << std::endl;
		return false;
	}

	vr::EVRInitError peError = vr::VRInitError_None;

	if(!vr::VRCompositor())
	{
		printf("Compositor initialization failed. See log file for details\n");
		return false;
	}

	m_pHMD->GetRecommendedRenderTargetSize(&m_RenderWidth, &m_RenderHeight);

	m_VRCamera.setup(*m_pHMD);

	return true;
}

bool RenderContext::handleSDL()
{
	SDL_Event sdlEvent;
	bool quitProgram = false;

	const auto currentFrameTime = SDL_GetPerformanceCounter();
	const auto deltaTime = ((currentFrameTime - m_LastFrameTime) * 1000 / static_cast<double>(SDL_GetPerformanceFrequency()));
	m_LastFrameTime = currentFrameTime;
	m_AccumulatedFrameTime += deltaTime;
	m_FrameCounter++;
	onFrameStart(deltaTime);

	auto cameraSpeedInSceneUnitPerMS = 0.01f;
	if(SDL_GetModState() & (SDL_Keymod::KMOD_LSHIFT | SDL_Keymod::KMOD_RSHIFT)) cameraSpeedInSceneUnitPerMS *= 10.f;

	while(SDL_PollEvent(&sdlEvent) != 0)
	{
		if(sdlEvent.type == SDL_QUIT)
		{
			quitProgram = true;
		}
		else if(sdlEvent.type == SDL_KEYDOWN)
		{
			onKeyPress(SDL_GetModState(), sdlEvent.key.keysym.sym);

			switch(sdlEvent.key.keysym.sym)
			{
			case SDLK_ESCAPE:
				quitProgram = true;
				break;
			case SDLK_w:
				if(m_pTrackballCamera)
					m_pTrackballCamera->move(-cameraSpeedInSceneUnitPerMS * static_cast<float>(deltaTime), 0.f);
				break;
			case SDLK_s:
				if(m_pTrackballCamera)
					m_pTrackballCamera->move(cameraSpeedInSceneUnitPerMS * static_cast<float>(deltaTime), 0.f);
				break;
			case SDLK_a:
				if(m_pTrackballCamera)
					m_pTrackballCamera->move(0.f, cameraSpeedInSceneUnitPerMS * static_cast<float>(deltaTime));
				break;
			case SDLK_d:
				if(m_pTrackballCamera)
					m_pTrackballCamera->move(0.f, -cameraSpeedInSceneUnitPerMS * static_cast<float>(deltaTime));
				break;
			case SDLK_e:
				m_RenderRightEye = !m_RenderRightEye;
				break;
			case SDLK_r:
				if(m_pTrackballCamera)
					m_pTrackballCamera->reset();
				break;
			case SDLK_t:
				if(m_FrameCounter)
				{
					std::cout << "Avg Frametime in ms over " << m_FrameCounter << " frames: " << m_AccumulatedFrameTime / m_FrameCounter << std::endl;
					m_AccumulatedFrameTime = 0;
					m_FrameCounter = 0;
				}
				break;
			case SDLK_i:
				PRINT_GL_INTEGER(GL_GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX);
				PRINT_GL_INTEGER(GL_GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX);
				PRINT_GL_INTEGER(GL_MAX_TEXTURE_SIZE);
				PRINT_GL_INTEGER(GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS);
				PRINT_GL_INTEGER(GL_MAX_GEOMETRY_OUTPUT_VERTICES);
				PRINT_GL_INTEGER(GL_MAX_GEOMETRY_TOTAL_OUTPUT_COMPONENTS);
				break;
			}
		}
		else if(sdlEvent.type == SDL_MOUSEBUTTONDOWN)
		{
			if(SDL_SetRelativeMouseMode(SDL_TRUE) != 0)
			{
				WARN("couldn't enable relative mouse mode" << SDL_GetError());
			}
		}
		else if(sdlEvent.type == SDL_MOUSEBUTTONUP)
		{
			if(SDL_SetRelativeMouseMode(SDL_FALSE) != 0)
			{
				WARN("couldn't enable relative mouse mode" << SDL_GetError());
			}
		}
		else if(sdlEvent.type == SDL_MOUSEMOTION)
		{
			auto motionEvent = sdlEvent.motion;
			auto buttonsHeld = motionEvent.state;
			if(buttonsHeld & SDL_BUTTON_LMASK)
			{
				if(m_pTrackballCamera)
					m_pTrackballCamera->rotate(glm::uvec2(m_RenderWidth / 2, m_RenderHeight / 2), glm::uvec2(m_RenderWidth / 2 + motionEvent.xrel, m_RenderHeight / 2 - motionEvent.yrel));
			}
			else if(buttonsHeld & SDL_BUTTON_RMASK)
			{
				if(m_pTrackballCamera)
					m_pTrackballCamera->pan(cameraSpeedInSceneUnitPerMS * motionEvent.xrel, cameraSpeedInSceneUnitPerMS * motionEvent.yrel);
			}
		}
		else if(sdlEvent.type == SDL_WINDOWEVENT &&
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
		printerr(__FILE__, __LINE__);
		renderStereoTargets();
		printerr(__FILE__, __LINE__);
		renderCompanionWindow();
		printerr(__FILE__, __LINE__);

		if(m_VREnabled)
		{
			vr::Texture_t leftEyeTexture = { (void*)(uintptr_t)m_LeftEyeFramebuffer.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::VRCompositor()->Submit(vr::Eye_Left, &leftEyeTexture);
			vr::Texture_t rightEyeTexture = { (void*)(uintptr_t)m_RightEyeFramebuffer.m_nResolveTextureId, vr::TextureType_OpenGL, vr::ColorSpace_Gamma };
			vr::VRCompositor()->Submit(vr::Eye_Right, &rightEyeTexture);
		}
	}

	if(m_bVblank)
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
	if(m_bVblank)
	{
		glFlush();
		glFinish();
	}

	printerr(__FILE__, __LINE__);
	if(m_VREnabled)
	{
		m_VRCamera.update();
	}
}

void RenderContext::renderQuad(vr::Hmd_Eye eye)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	auto vp = m_VRCamera.getMVP(eye);
	auto ivp = glm::inverse(vp);
	auto eyepos = m_VRCamera.getPosition(eye);
	if(!m_VREnabled)
	{
		vp = m_pTrackballCamera->projectionMatrix() * m_pTrackballCamera->viewMatrix();
		ivp = inverse(vp);
		eyepos = m_pTrackballCamera->getPosition();
	}

	onRenderEyeTexture(vp, eyepos);
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

	glUseProgram(m_DesktopProgram);

	// left eye
	glViewport(0, 0, m_nCompanionWindowWidth, m_nCompanionWindowHeight);
	glBindTexture(GL_TEXTURE_2D, m_LeftEyeFramebuffer.m_nResolveTextureId);
	Blit::instance().render();


	if(m_RenderRightEye)
	{
		// clear right eye space
		const unsigned int borderSize = 2u;
		glEnable(GL_SCISSOR_TEST);
		glScissor((m_nCompanionWindowWidth >> 2) + (m_nCompanionWindowWidth >> 1) - borderSize, 0, (m_nCompanionWindowWidth >> 2) + borderSize, borderSize + (m_nCompanionWindowHeight >> 2));
		glClearColor(0.1f, 0.4f, 0.8f, 1.f);
		glClear(GL_COLOR_BUFFER_BIT);
		glClearColor(0, 0, 0, 1);
		glDisable(GL_SCISSOR_TEST);

		// right eye
		glViewport((m_nCompanionWindowWidth >> 2) + (m_nCompanionWindowWidth >> 1), 0, m_nCompanionWindowWidth >> 2, m_nCompanionWindowHeight >> 2);
		glBindTexture(GL_TEXTURE_2D, m_LeftEyeFramebuffer.m_nResolveTextureId);
		Blit::instance().render();
	}

	glUseProgram(0);
}

void RenderContext::printGLInteger(const std::string& name, GLint constant)
{
	GLint x;
	glGetIntegerv(constant, &x);
	std::cout << name << ": " << x << std::endl;
}

void RenderContext::printerr(const char* file, const int line)
{
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("GLerror %s#%i: %i - %s\n", file, line, err, gluErrorString(err));
	}
}

void RenderContext::resize(int width, int height)
{
	m_nCompanionWindowWidth = width;
	m_nCompanionWindowHeight = height;
	WARN_TIMED("resizing window to " << width << "x" << height, 1)

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
	if(status != GL_FRAMEBUFFER_COMPLETE)
	{
		std::cout << "Framebuffer not complete" << std::endl;
		return false;
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	return true;
}

#undef PRINT_GL_INTEGER