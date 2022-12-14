// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"

#include <stdio.h>
#include <tchar.h>

#include <windows.h>
#include <iostream>
#include <memory>

#define _USE_MATH_DEFINES
#include <math.h>

#include <SDL.h>
#include <GL/glew.h>
#include <SDL_opengl.h>
#include <GL/glu.h>

//#define GLM_FORCE_CUDA 1
//#define GLM_FORCE_PURE 1
#define GLM_FORCE_SWIZZLE 1 
#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/glm.hpp>
#include <glm/gtc/type_ptr.hpp>

#include <future>

#include <chrono>
#include <time.h>

#include <LightFieldСontainer.h>
#include <SubdivisionSphere.h>
#include <SphereFace.h>
#include <SphereVertex.h>
#include <LightFieldData.h>
#include <RWRReader.h>

// prints msg
#define WARN( msg ) { std::cout << msg << std::endl; }
// print msg once
#define WARN_ONCE( msg )  { static bool warned = false; if(!warned){ warned=true; std::cout << msg << std::endl; } }
// print msg with a timeout in seconds
#define WARN_TIMED( msg, timeout ) { static time_t last_warned = 0; if ((last_warned + timeout) < time(0)) { last_warned = time(0);	std::cout << msg << std::endl; } }



// TODO: reference additional headers your program requires here
