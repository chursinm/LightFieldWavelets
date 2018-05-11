#include "stdafx.h"
#include "Texture.h"
#include <SDL_image.h>



Texture::Texture(const std::string & filename): m_Filename(filename), m_IsFinished(false), m_Width(0), m_Height(0), m_GLID(0)
{
	fetchFile();
}

Texture::Texture(const std::string & filename, GLuint glid) : m_Filename(filename), m_IsFinished(false), m_Width(0), m_Height(0), m_GLID(glid)
{
	fetchFile();
}

Texture::~Texture()
{
	glDeleteTextures(1, &m_GLID);
}

void Texture::fetchFile()
{
	m_SurfaceFuture = std::async(std::launch::async, [this]()
	{
		return IMG_Load(m_Filename.c_str());
	});
}

bool Texture::finished()
{
	return m_IsFinished;
}

void Texture::bind(unsigned int unit)
{
	glActiveTexture(GL_TEXTURE0 + unit);
	glBindTexture(GL_TEXTURE_2D, m_GLID);
}

void Texture::transferToGPU(SDL_Surface* surface)
{
	if(!surface)
	{
		WARN("Couldnt load surface " << m_Filename.c_str());
		return;
		//throw TextureException();
	}

	if(m_GLID == 0)
	{
		glGenTextures(1, &m_GLID);
	}
	glBindTexture(GL_TEXTURE_2D, m_GLID);

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

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	//glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

	m_Width = surface->w;
	m_Height = surface->h;

	SDL_FreeSurface(surface);

	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		WARN("Failed to upload texture to via opengl: " << gluErrorString(err));
		throw TextureException();
	}

	m_IsFinished = true;
	//WARN("Texture loaded at glid " << m_GLID);
}
