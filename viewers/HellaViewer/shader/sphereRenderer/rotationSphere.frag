#version 430 core

uniform vec3 viewspaceLightPosition;
uniform sampler2D debugTexture;
uniform int visualize_lightfield;

in vec3 viewspaceVertex;
in vec3 color;

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


vec3 phong(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 ambient, vec3 diffuse, vec3 specular, float shininess)
{
    float lambertian = max(dot(lightDir,normal), 0.0f);
    float specularExponent = 0.0f;

    if(lambertian > 0.0) {
        vec3 halfDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(halfDir, normal), 0.0f);
        specularExponent = pow(specAngle, shininess);
    }

    return vec3(ambient + diffuse * lambertian + specular * specularExponent);
}


void main(void)
{
	if(bool(visualize_lightfield))
	{
		outColor = vec4(color,0.5f);
	}
	else
	{
		int primitiveMod = gl_PrimitiveID % 12;
		vec3 color = COLORS[primitiveMod] * 0.7;

		vec3 viewDir = normalize(-viewspaceVertex);
		vec3 lightDir = normalize(viewspaceLightPosition - viewspaceVertex);
		vec3 normal = normalize(cross(dFdx(viewspaceVertex), dFdy(viewspaceVertex)));
		outColor = vec4(phong(viewDir, lightDir, normal, color*0.2, color, color, 0.5f), 1.0f);
		//outColor = texture(debugTexture, uv);
		//outColor = vec4(checkerboard(uv, 1));
		//outColor = vec4(uv, 0, 1);
	}
}
