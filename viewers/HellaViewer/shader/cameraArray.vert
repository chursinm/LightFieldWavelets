#version 400 compatibility
out vec2 uv;
uniform mat4 mvp;

void main(void)
{
    //uv = (gl_Vertex.xy + vec2(1,1)) * 0.5f;
	uv = gl_MultiTexCoord0.xy;
    gl_Position = mvp * gl_Vertex;
}