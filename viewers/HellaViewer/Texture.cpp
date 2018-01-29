#include "stdafx.h"
#include "Texture.h"
#include <SDL_image.h>


std::future<Texture> Texture::from(const std::string & path)
{
	return std::future<Texture>(std::async([]() { return Texture(); }));
}

Texture::Texture()
{
}


Texture::~Texture()
{
}
