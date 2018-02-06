#version 400 core

uniform mat4 ivp;

layout(location = 0) in vec2 clipspaceVertex;

noperspective out vec3 worldVertex3;

void main(void)
{
	vec4 clipspaceVertex4 = vec4(clipspaceVertex, 0.f, 1.f);
	vec4 worldVertex4 = ivp * clipspaceVertex4;
	worldVertex3 = worldVertex4.xyz / worldVertex4.w;
	gl_Position = clipspaceVertex4;
}