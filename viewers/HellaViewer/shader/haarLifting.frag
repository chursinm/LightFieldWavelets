#version 400 core

layout(location = 0) out vec4 outColor;

flat in float red;
flat in float blue;

void main(void)
{
	outColor = vec4(red,1.0f-red,blue,1);
}
