#pragma once
class Texture
{
public:
	Texture(const std::string& filename);
	//Texture(const Texture& ref);
	~Texture();
	
	void fetchFile();
	void asyncTransferToGPU(unsigned int waitForMS);

private:
	void transferToGPU(SDL_Surface* surface);
	std::string m_Filename;
	std::future<SDL_Surface*> m_SurfaceFuture;
	bool m_IsFinished;
	unsigned int m_Width, m_Height;
	GLuint m_GLID;
};

