#pragma once
class Texture
{
public:
	Texture();
	~Texture();
	static std::future<Texture> from(const std::string& path);
private:
	unsigned int m_Width, m_Height;
	GLuint m_GLID;
};

