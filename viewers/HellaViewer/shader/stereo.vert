#version 400 compatibility
//uniform mat4 ivp;
//noperspective out vec4 worldVertex;
noperspective out vec4 vertex;

void main(void)
{
	vertex = gl_Vertex;
	gl_Position = gl_Vertex;
}