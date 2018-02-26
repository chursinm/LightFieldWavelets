#pragma once
/*
	Simple utility class to render a fullscreen triangle
*/
class Blit
{
public:
	static Blit& instance()
	{
		static Blit _instance;
		return _instance;
	}
	/*
		Renders the fullscreen triangle. Don't forget to bind your program beforehand!
	*/
	void render();
private:
	void generateTriangle();
	GLuint m_vao, m_vb;
	Blit();
	Blit(const Blit&);
	Blit & operator = (const Blit&);
	~Blit();
};

