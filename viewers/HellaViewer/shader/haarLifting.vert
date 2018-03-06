#version 410 core

uniform mat4 mvp;
uniform float scale;

layout(location = 0) in float xCoord;
layout(location = 1) in float yCoord;

flat out float red;
flat out float blue;

void main(void)
{
	red = float(int(xCoord) % 2);
	blue = float(int(xCoord) % 10 > 5);
	gl_Position = mvp * vec4(xCoord * scale, yCoord * scale,-1, 1);
}