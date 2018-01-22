#version 400 compatibility
in vec2 uv;

void main(void)
{
    gl_FragColor = vec4(uv,0,0);
}
