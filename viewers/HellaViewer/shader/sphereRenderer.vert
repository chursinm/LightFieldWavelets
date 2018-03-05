#version 410 core

uniform mat4 mvp;
uniform mat4 viewMatrix;

layout(location = 0) in vec3 vertex;

out vec3 vertViewspaceVertex;

void main(void)
{
	vec4 viewspaceVertex4 = viewMatrix * vec4(vertex, 1.0f);
	vertViewspaceVertex = viewspaceVertex4.xyz / viewspaceVertex4.w;
	gl_Position = mvp * vec4(vertex, 1.0f);
}