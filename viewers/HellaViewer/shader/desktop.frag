#version 400 core
uniform sampler2D renderedImage;

noperspective in vec2 uv;

layout(location = 0) out vec4 outColor;

void main(void)
{
	outColor = texture(renderedImage, uv);
}
