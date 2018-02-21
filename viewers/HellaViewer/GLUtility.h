#pragma once
class GLUtility
{
public:
	template<class T>
	static GLuint generateBuffer(GLenum target, unsigned int elements, const T* data, GLenum usage)
	{
		GLuint id;
		glGenBuffers(1, &id);
		glBindBuffer(target, id);
		glBufferData(target, elements * sizeof(T), data, usage);
		glBindBuffer(target, 0);
		return id;
	}
	template<class T>
	static GLuint generateBuffer(GLenum target, std::vector<T> data, GLenum usage)
	{
		return generateBuffer(target, data.size(), &data[0], usage);
	}
private:
	GLUtility();
	~GLUtility();
};
