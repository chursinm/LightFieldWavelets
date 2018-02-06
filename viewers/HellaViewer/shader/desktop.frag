#version 400 core
uniform sampler2D renderedImage;

in vec2 uv;

void main(void)
{
    //gl_FragColor = vec4(uv,0,0);
	gl_FragColor = texture(renderedImage, uv);
}
