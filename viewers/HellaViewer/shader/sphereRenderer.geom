#version 460 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec2 vertUv[];
in vec3 vertViewspaceVertex[];

out vec3 edgeDistance;
out vec3 viewspaceVertex;
out vec2 uv;
out int gl_PrimitiveID;

void main()
{
	edgeDistance = vec3(1,0,0);
	gl_Position = gl_in[0].gl_Position;
	viewspaceVertex = vertViewspaceVertex[0];
	uv = vertUv[0];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	
	edgeDistance = vec3(0,1,0);
	gl_Position = gl_in[1].gl_Position;
	viewspaceVertex = vertViewspaceVertex[1];
	uv = vertUv[1];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	
	edgeDistance = vec3(0,0,1);
	gl_Position = gl_in[2].gl_Position;
	viewspaceVertex = vertViewspaceVertex[2];
	uv = vertUv[2];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	

    //EndPrimitive();
}