#version 400 compatibility
out vec2 uv;
uniform mat4 mvp;
noperspective out vec4 vertex;

void main(void)
{
    //uv = (gl_Vertex.xz * 0.05f + .5f);
    //gl_Position = mvp * gl_Vertex;
	uv = (gl_Vertex.xy * .5f + .5f);
	vertex = gl_Vertex;
	gl_Position = gl_Vertex;
}