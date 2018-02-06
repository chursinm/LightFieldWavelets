#version 400 core
uniform sampler2D renderedImage;

in vec2 uv;

layout(location = 0) out vec4 outColor;

void main(void)
{
    //gl_FragColor = vec4(uv,0,0);
	outColor = texture(renderedImage, uv);
}
