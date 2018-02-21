#version 400 core
layout(location = 0) in vec2 clipspaceVertex;
layout(location = 1) in vec2 uvIn;

noperspective out vec2 uv;

void main(void)
{
	uv = uvIn;
    gl_Position = vec4(clipspaceVertex, 0.f, 1.f);
}