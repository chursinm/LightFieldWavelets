#version 400 compatibility

in vec2 uv;
uniform sampler2D cameraImage;

void main(void)
{
    gl_FragColor = vec4(uv,0.f,0.f);
	//gl_FragColor = texture(cameraImage, uv);
	//gl_FragColor = vec4(1,0,0,0);
}
