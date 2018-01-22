#include "stdafx.h"
#include "ShaderManager.h"
#include <fstream>
#include <vector>

GLuint ShaderManager::from(const std::string& vertexShaderFile, const std::string& fragmentShaderFile)
{
	auto it = mProgramCache.find({vertexShaderFile, fragmentShaderFile});
	if (it != mProgramCache.end())
	{
		return it->second;
	}

	GLuint vertexShaderID = vertexShader(vertexShaderFile);
	GLuint fragmentShaderID = fragmentShader(fragmentShaderFile);
	if (vertexShaderID == 0 || fragmentShaderID == 0)
	{
		return 0;
	}

	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Folgende Zeilen stammen von http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
	// Link the program
	fprintf(stdout, "Linking program\n");
	GLuint program = glCreateProgram();
	glAttachShader(program, vertexShaderID);
	glAttachShader(program, fragmentShaderID);
	glLinkProgram(program);

	// Check the program
	glGetProgramiv(program, GL_LINK_STATUS, &Result);
	glGetProgramiv(program, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if (InfoLogLength > 0)
	{
		std::vector<char> ProgramErrorMessage(InfoLogLength);
		glGetProgramInfoLog(program, InfoLogLength, NULL, &ProgramErrorMessage[0]);
		fprintf(stdout, "%s\n", &ProgramErrorMessage[0]);
	}

	// Diese werden in anderen Programmen wiederverwendet
	//glDeleteShader(m_VertexShader);
	//glDeleteShader(m_FragmentShader);

	if (Result != GL_FALSE)
	{
		mProgramCache.emplace(std::pair<std::pair<std::string, std::string>, GLuint>({ vertexShaderFile, fragmentShaderFile }, program));
		return program;
	}
	else
	{
		glDeleteProgram(program);
		return 0;
	}
}

GLuint ShaderManager::vertexShader(const std::string& filename)
{
	auto it = mVertexShaderCache.find(filename);
	if (it != mVertexShaderCache.end())
	{
		return it->second;
	}

	GLuint shader = loadShader(filename, GL_VERTEX_SHADER);

	if(shader != 0) mVertexShaderCache.emplace(std::pair<std::string, GLuint>(filename, shader));
	return shader;
}

GLuint ShaderManager::fragmentShader(const std::string& filename)
{
	auto it = mFragmentShaderCache.find(filename);
	if (it != mFragmentShaderCache.end())
	{
		return it->second;
	}

	GLuint shader = loadShader(filename, GL_FRAGMENT_SHADER);

	if(shader != 0) mFragmentShaderCache.emplace(std::pair<std::string, GLuint>(filename, shader));
	return shader;
}

GLuint ShaderManager::loadShader(const std::string& filename, int shaderType)
{
	
	// Folgende Zeilen stammen von http://www.opengl-tutorial.org/beginners-tutorials/tutorial-2-the-first-triangle/
	// Read the Vertex Shader code from the file
	std::string shaderCode;
	std::ifstream shaderStream(filename, std::ifstream::in);
	if (shaderStream.is_open())
	{
		std::string Line = "";
		while (getline(shaderStream, Line))
			shaderCode += "\n" + Line;
		shaderStream.close();
	}
	else
	{
		std::cout << "Couldn't open shader file" << std::endl;
		return 0;
	}

	GLuint shaderID = glCreateShader(shaderType);
	GLint Result = GL_FALSE;
	int InfoLogLength;

	// Compile Vertex Shader
	std::cout << "Compiling shader: " << filename << std::endl;
	char const * VertexSourcePointer = shaderCode.c_str();
	glShaderSource(shaderID, 1, &VertexSourcePointer, NULL);
	glCompileShader(shaderID);

	// Check Vertex Shader
	glGetShaderiv(shaderID, GL_COMPILE_STATUS, &Result);
	glGetShaderiv(shaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
	if(InfoLogLength >0)
	{
		std::vector<char> VertexShaderErrorMessage(InfoLogLength);
		glGetShaderInfoLog(shaderID, InfoLogLength, NULL, &VertexShaderErrorMessage[0]);
		fprintf(stdout, "%s\n", &VertexShaderErrorMessage[0]);
	}

	if (Result != GL_FALSE)
	{
		return shaderID;
	}
	else
	{
		glDeleteShader(shaderID);
		return 0;
	}
}

ShaderManager::ShaderManager()
{
}


ShaderManager::~ShaderManager()
{
	for(auto& it : mVertexShaderCache)
	{
		glDeleteShader(it.second);
	}
	for (auto& it : mFragmentShaderCache)
	{
		glDeleteShader(it.second);
	}
	for (auto& it : mProgramCache)
	{
		glDeleteProgram(it.second);
	}
}
