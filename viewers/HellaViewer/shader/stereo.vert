#version 400 compatibility
out vec2 uv;
uniform mat4 mvp;

void main(void)
{
    uv = (gl_Vertex.xz * 0.05f + .5f);
    gl_Position = mvp * gl_Vertex;
}