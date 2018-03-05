#version 410 core
#define PI 3.1415926535897932384626433832795

uniform mat4 mvp;
uniform mat4 viewMatrix;

layout(location = 0) in vec3 vertex;

out vec3 vertViewspaceVertex;
out vec2 vertUv;

void main(void)
{
	vec3 test = normalize(vertex);
	// Rotation Y Axis (A)
	vec2 yRotVert = test.xz; //normalize(vertex.xz);
	vertUv.y = (atan(yRotVert.x, yRotVert.y) / PI) * 0.5 + 0.5;
	// Rotation X Axis (b)
	vec2 xRotVert = test.yz; //normalize(vertex.yz);
	vertUv.x = acos(xRotVert.x) / PI;



	vec4 viewspaceVertex4 = viewMatrix * vec4(vertex, 1.0f);
	vertViewspaceVertex = viewspaceVertex4.xyz / viewspaceVertex4.w;
	gl_Position = mvp * vec4(vertex, 1.0f);
}