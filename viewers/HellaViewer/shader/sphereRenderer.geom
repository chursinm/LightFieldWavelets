#version 460 core

layout(triangles) in;
layout(triangle_strip, max_vertices=3) out;

in vec3 vertViewspaceVertex[];

out vec3 edgeDistance;
out vec3 viewspaceVertex;
out int gl_PrimitiveID;

void main()
{
	edgeDistance = vec3(1,0,0);
	gl_Position = gl_in[0].gl_Position;
	viewspaceVertex = vertViewspaceVertex[0];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	
	edgeDistance = vec3(0,1,0);
	gl_Position = gl_in[1].gl_Position;
	viewspaceVertex = vertViewspaceVertex[1];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	
	edgeDistance = vec3(0,0,1);
	gl_Position = gl_in[2].gl_Position;
	viewspaceVertex = vertViewspaceVertex[2];
	gl_PrimitiveID = gl_PrimitiveIDIn;
	EmitVertex();
	

    //EndPrimitive();
}