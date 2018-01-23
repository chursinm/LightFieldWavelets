#version 400 compatibility
noperspective out vec4 vertex;

void main(void)
{
	vertex = gl_Vertex;
	gl_Position = gl_Vertex;
}