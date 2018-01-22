// HellaViewer.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "RenderContext.h"

int main(int argc, char** argv)
{
	bool quit = false;

	RenderContext context;
	if (!context.initializeSDL()) { return -1; }
	if (!context.initializeGL()) { return -1; }
	
	while (!quit)
	{
		quit = context.handleSDL();
		context.render();
	}

    return 0;
}
