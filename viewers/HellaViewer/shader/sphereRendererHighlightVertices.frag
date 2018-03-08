#version 400 core

uniform vec3 viewspaceLightPosition;
uniform sampler2D debugTexture;
uniform float pointSize;

in vec3 viewspaceVertex;
in vec2 uv;

layout(location = 0) out vec4 outColor;

// constant as uniform for random performance reasons
uniform vec3 COLORS[ 12 ] = {	vec3( 1.0, 0.0, 0.0 ),
                                vec3( 0.0, 1.0, 0.0 ),
                                vec3( 0.0, 0.0, 1.0 ),
                                vec3( 1.0, 1.0, 0.0 ),
                                vec3( 0.0, 1.0, 1.0 ),
                                vec3( 1.0, 0.0, 1.0 ),

								vec3( 1.0, 0.4, 0.4 ),
                                vec3( 0.4, 1.0, 0.4 ),
                                vec3( 0.4, 0.4, 1.0 ),
                                vec3( 1.0, 1.0, 0.4 ),
                                vec3( 0.4, 1.0, 1.0 ),
                                vec3( 1.0, 0.4, 1.0 )  };


bool checkerboard(vec2 a, int level)
{
	return mod(a.x*level*2, 2) >= 1 != mod(a.y*level*2, 2) >= 1;
}

void main(void)
{
	vec2 circCoord = 2.0 * gl_PointCoord - 1.0;
	if (dot(circCoord, circCoord) > 1.0) {
		discard;
	}

	int primitiveMod = gl_PrimitiveID % 12;
	vec3 color = COLORS[primitiveMod] * 0.7;

	//outColor = vec4(color, 1.0);
	outColor = vec4(texture(debugTexture, gl_PointCoord).rgb, 0.4);
	//outColor = vec4(checkerboard(uv, 1));
	//outColor = vec4(uv, 0, 1);
}
