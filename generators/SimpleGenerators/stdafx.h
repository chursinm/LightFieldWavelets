// stdafx.h : include file for standard system include files,
// or project specific include files that are used frequently, but
// are changed infrequently
//

#pragma once

#include "targetver.h"
#include <stdio.h>
#include <tchar.h>

#include <windows.h>
#include <memory>
#include <iostream>
#include <sstream>
#include <string>
#include <climits>
#include <vector>

#define _USE_MATH_DEFINES
#include <math.h>
#include <algorithm>

#define GLM_FORCE_SWIZZLE 1 
#define GLM_ENABLE_EXPERIMENTAL 1
#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>
#define SDL_MAIN_HANDLED // as we do not actually want sdl to handle anything
#include <SDL.h>
#include <SDL_image.h>

#include <SphereFace.h>
#include <SphereVertex.h>
#include <RayExtractor.h>

#include <LightFieldData.h>
#include <SubdivisionSphere.h>






// TODO: reference additional headers your program requires here
