#version 410 core
#define PI 3.1415926535897932384626433832795

uniform mat4 mvp;
uniform mat4 viewMatrix;
uniform float pointSize;

layout(location = 0) in vec3 vertex;

out vec3 viewspaceVertex;
out vec2 uv;

void main(void)
{
	vec3 test = normalize(vertex);
	// Rotation Y Axis (A)
	vec2 yRotVert = test.xz; //normalize(vertex.xz);
	uv.y = (atan(yRotVert.x, yRotVert.y) / PI) * 0.5 + 0.5;
	// Rotation X Axis (b)
	vec2 xRotVert = test.yz; //normalize(vertex.yz);
	uv.x = acos(xRotVert.x) / PI;

	vec4 viewspaceVertex4 = viewMatrix * vec4(vertex, 1.0f);
	viewspaceVertex = viewspaceVertex4.xyz / viewspaceVertex4.w;
	gl_Position = mvp * vec4(vertex, 1.0f);
	gl_PointSize = pointSize;
}