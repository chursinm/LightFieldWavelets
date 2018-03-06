#pragma once
class GLUtility
{
public:
	static GLuint generateBuffer(GLenum target, unsigned long long bytesize, GLenum usage)
	{
		GLuint id;
		glGenBuffers(1, &id);
		glBindBuffer(target, id);
		glBufferData(target, bytesize, nullptr, usage);
		if(id == 0) throw "failed to generate gl buffer";
		return id;
	}
	template<class T>
	static GLuint generateBuffer(GLenum target, unsigned int elements, const T* data, GLenum usage)
	{
		GLuint id;
		glGenBuffers(1, &id);
		glBindBuffer(target, id);
		glBufferData(target, elements * sizeof(T), data, usage);
		if(id == 0) throw "failed to generate gl buffer";
		return id;
	}
	template<class T>
	static GLuint generateBuffer(GLenum target, const std::vector<T>& data, GLenum usage)
	{
		return generateBuffer(target, data.size(), &data[0], usage);
	}


#pragma region uniformHandling
	static GLint getUniformId(const GLuint programId, const std::string& key)
	{
		return glGetUniformLocation(programId, key.c_str());
	}
	static void setUniform(const GLuint programId, const std::string& key, const glm::vec3& value)
	{
		glUniform3fv(getUniformId(programId, key), 1, &value[0]);
	}
	static void setUniform(const GLuint programId, const std::string& key, const glm::mat4x4& value)
	{
		glUniformMatrix4fv(getUniformId(programId, key), 1, GL_FALSE, &value[0][0]);
	}
	static void setUniform(const GLuint programId, const std::string& key, const float& value)
	{
		glUniform1f(getUniformId(programId, key), value);
	}
	static void setUniform(const GLuint programId, const std::string& key, const int& value)
	{
		glUniform1i(getUniformId(programId, key), value);
	}
	static void setUniform(const GLuint programId, const std::string& key, const unsigned int& value)
	{
		glUniform1ui(getUniformId(programId, key), value);
	}
#pragma endregion

private:
	GLUtility();
	~GLUtility();
};
