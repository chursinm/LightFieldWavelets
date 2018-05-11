#version 430 core
#define PI 3.1415926535897932384626433832795

uniform mat4 mvp;
uniform mat4 viewMatrix;
uniform float scaleFac;
uniform int vertexCount;

layout(location = 0) in vec3 vertexIn;
layout(location = 2) in vec3 instancedPosition;
layout(std430, binding = 3) buffer Lightfield
{
	vec4 data[];
};

out vec3 viewspaceVertex;
out vec3 color;

void main(void)
{
	vec3 vertex = vertexIn * scaleFac + instancedPosition;
	vec4 viewspaceVertex4 = viewMatrix * vec4(vertex, 1.0f);
	viewspaceVertex = viewspaceVertex4.xyz / viewspaceVertex4.w;
	color = data[gl_InstanceID * vertexCount + gl_VertexID].rgb;
	gl_Position = mvp * vec4(vertex, 1.0f);
}