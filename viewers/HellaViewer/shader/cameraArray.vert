#version 460 core

uniform mat4 mvp;
uniform uvec2 cameraGridDimension;
uniform uint quadID;
uniform mat4 ivp;
uniform vec3 worldspaceFocalPlanePosition;
uniform vec3 worldspaceFocalPlaneDirection;
uniform vec3 worldspaceEyePosition;

layout(location = 0) in vec3 worldspaceCameraplaneVertex;

out vec2 uv;
sample out vec4 clipspaceCameraplaneVertex;
flat out vec4 clipspaceCameraplaneCamera;

void main(void)
{
	//quadID = gl_DrawID;

	uint gridCameraWidth = cameraGridDimension.x;
	uint gridQuadWidth = cameraGridDimension.x - 1u;
	
	uint quadX = quadID % gridQuadWidth;
	uint quadY = quadID / gridQuadWidth;

	uint cameraX = gl_VertexID % gridCameraWidth;
	uint cameraY = gl_VertexID / gridCameraWidth;

	uv = vec2(cameraX - quadX, cameraY - quadY);

    //uv = (gl_Vertex.xy + vec2(1,1)) * 0.5f;
	//uv = gl_MultiTexCoord0.xy;

	clipspaceCameraplaneVertex = mvp * vec4(worldspaceCameraplaneVertex, 1.f);
	clipspaceCameraplaneCamera = clipspaceCameraplaneVertex;
    gl_Position = clipspaceCameraplaneVertex;
}