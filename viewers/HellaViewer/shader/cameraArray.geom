#version 460 core

layout(triangles) in;
layout(triangle_strip) out;
layout(max_vertices=3) out;

in vec2 vert_uv[];
sample in vec4 vert_clipspaceCameraplaneVertex[];
flat in vec4 vert_clipspaceCameraplaneCamera[];

out vec2 uv;
sample out vec4 clipspaceCameraplaneVertex;
flat out vec4 clipspaceCameraplaneCamera;

flat out vec4 clipspaceCameraplaneCameraArray[3];


void main()
{
	for (int i = 0; i < 3; i++) {
		gl_Position = gl_in[i].gl_Position;
		uv = vert_uv[i];
		clipspaceCameraplaneVertex = vert_clipspaceCameraplaneVertex[i];
		clipspaceCameraplaneCamera = vert_clipspaceCameraplaneCamera[i];
		clipspaceCameraplaneCameraArray[0] = vert_clipspaceCameraplaneCamera[0];
		clipspaceCameraplaneCameraArray[1] = vert_clipspaceCameraplaneCamera[1];
		clipspaceCameraplaneCameraArray[2] = vert_clipspaceCameraplaneCamera[2];
		EmitVertex();
	}
}