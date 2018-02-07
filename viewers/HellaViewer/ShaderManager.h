#pragma once
#include <map>
#include <string>

/**
	@author Christoph Rohmann

	Eine einfache globale Verwaltung von OpenGL Programs.
	Läd GLSL Shader aus angegebenen Dateien und baut aus diesen GLPrograms.
	Inkombatibel mit mehreren GL Kontexten.
*/
class ShaderManager
{
public:
	static ShaderManager& instance()
	{
		static ShaderManager _instance;
		return _instance;
	}
	GLuint from(const std::string& vertexShaderFile, const std::string& fragmentShaderFile);
	GLuint from(const std::string& vertexShaderFile, const std::string& geometryShaderFile, const std::string& fragmentShaderFile);
	virtual ~ShaderManager();
private:
	ShaderManager();
	ShaderManager(const ShaderManager&); 
	ShaderManager & operator = (const ShaderManager&);

	std::map<std::string, GLuint> mVertexShaderCache;
	std::map<std::string, GLuint> mGeometryShaderCache;
	std::map<std::string, GLuint> mFragmentShaderCache;
	std::map<std::pair<std::string, std::string>, GLuint> mProgramCache;


	GLuint vertexShader(const std::string& filename);
	GLuint geometryShader(const std::string& filename);
	GLuint fragmentShader(const std::string& filename);
	GLuint loadShader(const std::string& filename, int shaderType);

	static ShaderManager* s_pSingletonInstance;
};

