#include "stdafx.h"
#include "Texture.h"
#include <SDL_image.h>



Texture::Texture(const std::string & filename): m_Filename(filename), m_IsFinished(false), m_Width(0), m_Height(0), m_GLID(0)
{
}

//Texture::Texture(const Texture & ref): m_Filename(ref.m_Filename), m_SurfaceFuture(), m_IsFinished(ref.m_IsFinished), m_Width(ref.m_Width), m_Height(ref.m_Height), m_GLID(ref.m_GLID)
//{
//}

Texture::~Texture()
{
}

void Texture::fetchFile()
{
	m_SurfaceFuture = std::async(std::launch::async, [this]()
	{
		return IMG_Load(m_Filename.c_str());
	});
}

void Texture::asyncTransferToGPU(unsigned int waitForMS)
{
	if(m_IsFinished) { return; }
	if(!m_SurfaceFuture.valid())
	{
		WARN_TIMED("Invalid future, was fetchFile() called?", 2)
		return;
	}

	{
		auto status = m_SurfaceFuture.wait_for(std::chrono::milliseconds(waitForMS));
		if(status == std::future_status::deferred)
		{
			std::cout << "deferred\n";
		}
		else if(status == std::future_status::timeout)
		{
			std::cout << "timeout\n";
		}
		else if(status == std::future_status::ready)
		{
			transferToGPU(m_SurfaceFuture.get());
		}
	}

}

void Texture::transferToGPU(SDL_Surface* surface)
{
	if(!surface)
	{
		WARN("Couldnt load surface " << m_Filename.c_str());
		return;
	}

	GLuint TextureID = 0;
	glGenTextures(1, &TextureID);
	glBindTexture(GL_TEXTURE_2D, TextureID);

	auto Mode = GL_RGB;

	if(surface->format->BytesPerPixel == 4)
	{
		Mode = GL_RGBA;
	}

	auto num_mipmaps = 1u;
	glTexStorage2D(GL_TEXTURE_2D, num_mipmaps, GL_RGB8, surface->w, surface->h);
	glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, surface->w, surface->h, Mode, GL_UNSIGNED_BYTE, surface->pixels);
	//glTexImage2D(GL_TEXTURE_2D, 0, Mode, Surface->w, Surface->h, 0, Mode, GL_UNSIGNED_BYTE, Surface->pixels);

	glGenerateMipmap(GL_TEXTURE_2D);  //Generate num_mipmaps number of mipmaps here.

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	m_Width = surface->w;
	m_Height = surface->h;
	m_IsFinished = true;
	m_GLID = TextureID;

	SDL_FreeSurface(surface);

	WARN_ONCE("Texture loaded" << TextureID)
}
