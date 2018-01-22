#version 400 compatibility
noperspective out vec2 uv;

void main(void)
{
    //uv = (gl_Vertex.xy + vec2(1,1)) * 0.5f;
	uv = gl_MultiTexCoord0.xy;
    gl_Position = gl_Vertex;
}