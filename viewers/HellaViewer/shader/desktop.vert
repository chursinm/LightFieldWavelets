#version 400 core
layout(location = 0) in vec2 clipspaceVertex;

noperspective out vec2 uv;

void main(void)
{
    uv = (clipspaceVertex + vec2(1.f,1.f)) * 0.5f;
    gl_Position = vec4(clipspaceVertex, 0.f, 1.f);
}