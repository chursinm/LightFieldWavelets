#version 400 compatibility
out vec2 uv;

void main(void)
{
    uv = (gl_Vertex.xy + vec2(1,1)) * 0.5f;
    gl_Position = gl_Vertex;
}