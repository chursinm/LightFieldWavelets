#version 460 compatibility

uniform mat4 mvp;
uniform uvec2 cameraGridDimension;
uniform uint quadID;

out vec2 uv;
out vec4 clipspaceCameraplaneVertex;
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

	clipspaceCameraplaneVertex = mvp * gl_Vertex;
	clipspaceCameraplaneCamera = clipspaceCameraplaneVertex;
    gl_Position = clipspaceCameraplaneVertex;
}