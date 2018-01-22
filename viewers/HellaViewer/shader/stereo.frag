#version 400 compatibility
in vec2 uv;
uniform uint eye;

void main(void)
{
	gl_FragColor = vec4(uv,0,0);
}
